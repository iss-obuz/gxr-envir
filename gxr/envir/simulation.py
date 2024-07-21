from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from more_itertools import chunked
from pqdm.processes import pqdm
from pyarrow.parquet import ParquetWriter
from tqdm.auto import tqdm

from gxr.dotpath import dotget, dotset
from gxr.envir.config import Config
from gxr.envir.game import EnvirGame
from gxr.envir.model import EnvirModel


class Simulator:
    """Simulation runner for a single configuration of parameters.

    Attributes
    ----------
    fields
        Parameter and data fields to extract.
        Defined as a mapping from field names to dotpaths.
    **kwargs
        Passed to :meth:`gxr.envir.game.EnvirGame`.
    """

    def __init__(
        self,
        fields: Mapping[str, str],
        overrides: Mapping[str, Any] | None = None,
        **game_kwargs: Any,
    ) -> None:
        self.fields = fields
        self.overrides = overrides or {}
        self.config = Config(resolve=False, interpolate=False)
        self.game_kwargs = game_kwargs

        for dotpath, value in self.overrides.items():
            dotset(self.config, dotpath, value, item=True)

    def make_game(
        self,
        random_state: np.random.Generator | None = None,
    ) -> EnvirGame:
        config = Config(self.config, resolve=True, interpolate=True)
        model = EnvirModel(**config["model"])

        if random_state is not None:
            model.behavior.rng = random_state
        return EnvirGame(model, **self.game_kwargs)

    def run(
        self,
        n_epochs: float,
        n_reps: int,
        n_jobs: int = 1,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        records = []
        n_jobs = min(n_jobs, n_reps)
        fields = self.extract_fields()

        if not isinstance(random_state, np.random.Generator):
            random_state = np.random.default_rng(np.random.PCG64(random_state))

        records = []

        for rng in random_state.spawn(n_reps):
            game = self.make_game(rng)
            sol = game.dynamics.run(n_epochs, **kwargs)
            results = game.get_results(sol)
            record = fields.copy()
            for field, value in results.items():
                if field in record:
                    errmsg = f"duplicated '{field}' field"
                    raise KeyError(errmsg)
                record[field] = value
            records.append(record)

        data = pd.DataFrame(records)
        return data.convert_dtypes(convert_integer=False)

    def extract_fields(self) -> Mapping[str, Any]:
        """Extract fields from a ``game`` instance."""
        return {
            field: dotget(self.config, dotpath, item=True)
            for field, dotpath in self.fields.items()
        }


class Simulation:
    """Simulation experiment class for exploring parameter spaces.

    Attributes
    ----------
    n_epochs
        Number of epochs to run the simulation for.
    n_reps
        Number of independent repetitions for a given configuration.
    **kwargs
        Passed to :meth:`gxr.envir.game.EnvirGame`.
    """

    def __init__(
        self,
        n_epochs: float,
        n_reps: int,
        params: Mapping[str, Sequence[Any]],
        fields: Mapping[str, str] | None = None,
        batch_scale: int = 5,
        seed: int | None = None,
        *,
        n_jobs: int = 1,
        **game_kwargs: Any,
    ) -> None:
        self.n_epochs = n_epochs
        self.n_reps = n_reps
        self.params = {p: np.array(v).tolist() for p, v in params.items()}
        self.fields = fields or {k.rsplit(".", 1)[-1]: k for k in params}
        self.batch_scale = batch_scale
        self.seed_sequence = np.random.SeedSequence(seed)
        self.n_jobs = n_jobs
        self.game_kwargs = game_kwargs

    @property
    def n_points(self) -> int:
        return np.prod(list(map(len, self.params.values())))

    @property
    def n_batches(self) -> int:
        return int(np.ceil(self.n_points / self.batch_size))

    @property
    def batch_size(self) -> int:
        return self.n_jobs * self.batch_scale

    def iter_simulators(self) -> Iterator[Simulator]:
        points = product(*self.params.values())
        for point in points:
            overrides = dict(zip(self.params, point, strict=True))
            simulator = Simulator(self.fields, overrides=overrides, **self.game_kwargs)
            yield simulator

    def iter_batches(self) -> Iterator[Iterable[Simulator]]:
        yield from chunked(self.iter_simulators(), n=self.batch_size)

    def run_batch(self, batch: Iterable[Simulator]) -> pd.DataFrame:
        batch = list(batch)
        seeds = self.seed_sequence.generate_state(len(batch))
        args = zip(batch, seeds, strict=True)
        results = pqdm(
            list(args),
            self.compute,
            n_jobs=self.n_jobs,
            leave=False,
        )
        data = pd.concat(results, axis=0, ignore_index=True).convert_dtypes()
        return data.sort_values(by=list(self.fields))

    def iter_results(
        self,
        *,
        progress: bool = True,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame]:
        if n_jobs is not None:
            self.n_jobs = n_jobs
        tqdm_kwargs = dict(total=self.n_batches, disable=not progress, **kwargs)
        for batch in tqdm(self.iter_batches(), **tqdm_kwargs):
            yield self.run_batch(batch)

    def compute(self, args: tuple[Simulator, np.random.Generator]) -> pd.DataFrame:
        simulator, seed = args
        return simulator.run(
            n_epochs=self.n_epochs, n_reps=self.n_reps, random_state=seed
        )

    def run(
        self, path: str | Path | None = None, compression: str = "zstd", **kwargs: Any
    ) -> None | pd.DataFrame:
        parts = self.iter_results(**kwargs)

        if not path:
            return pd.concat(parts, axis=0, ignore_index=True).sort_values(
                by=list(self.fields)
            )

        path = Path(path)
        writer = None
        exception = None
        try:
            for part in parts:
                data = pl.from_pandas(part).to_arrow()
                if writer is None:
                    writer = ParquetWriter(path, data.schema, compression=compression)
                writer.write_table(data)
        except Exception as exc:
            exception = exc
        finally:
            if writer:
                writer.close()
        if exception:
            raise exception
        return None
