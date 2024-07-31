from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any, Self

import joblib
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
        split_epochs: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        records = []
        n_jobs = min(n_jobs, n_reps)
        fields = self.extract_fields()

        if split_epochs and round(n_epochs) != n_epochs:
            errmsg = "'n_epochs' must be an integer when 'split_epochs=True'"
            raise ValueError(errmsg)

        if not isinstance(random_state, np.random.Generator):
            random_state = np.random.default_rng(np.random.PCG64(random_state))

        records = []

        for idx, rng in enumerate(random_state.spawn(n_reps), 1):
            game = self.make_game(rng)

            epochs = [1] * int(n_epochs) if split_epochs else [n_epochs]

            for dt in epochs:
                sol = game.dynamics.run(dt, **kwargs)
                results = {"rep_id": idx, **game.get_results(sol)}
                results["R"] = results["U"].reshape(game.n_agents, -1).mean(0)
                record = fields.copy()

                sim_id = joblib.hash(
                    (fields["param_id"], results["rep_id"]), hash_name="md5"
                )
                results = {"sim_id": sim_id, **results}

                for field, value in results.items():
                    if field in record:
                        errmsg = f"duplicated '{field}' field"
                        raise KeyError(errmsg)
                    record[field] = value

                if split_epochs:
                    record["epoch"] = int(record["epochs"][-1])

                records.append(record)

        data = pd.DataFrame(records)
        return data.convert_dtypes(convert_integer=False)

    def extract_fields(self) -> Mapping[str, Any]:
        """Extract fields from a ``game`` instance."""
        fields = {
            field: dotget(self.config, dotpath, item=True)
            for field, dotpath in self.fields.items()
        }
        idx = joblib.hash(tuple(map(tuple, fields.items())), hash_name="md5")
        return {**fields, "param_id": idx}


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
        split_epochs: bool = True,
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
        self.split_epochs = split_epochs

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
        if not self.split_epochs:
            del data["epoch"]
        df = data.sort_values(by=list(self.fields))
        return SimulationFrame(df)

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
            n_epochs=self.n_epochs,
            n_reps=self.n_reps,
            random_state=seed,
            split_epochs=self.split_epochs,
        )

    def run(
        self, path: str | Path | None = None, compression: str = "zstd", **kwargs: Any
    ) -> None | pd.DataFrame:
        parts = self.iter_results(**kwargs)

        if not path:
            df = pd.concat(parts, axis=0, ignore_index=True).sort_values(
                by=list(self.fields)
            )
            return SimulationFrame(df)

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


class SimulationFrame(pd.DataFrame):
    @property
    def _constructor(self) -> type["SimulationFrame"]:
        return self.__class__

    def group_epochs(self) -> Self:
        if "epoch" not in self.columns:
            return self
        array_cols = ["epochs", "T", "E", "H", "P", "U", "R"]
        other_cols = [c for c in self.columns if c not in array_cols]
        df = (
            self.groupby(["sim_id"])
            .agg(
                {
                    **{k: lambda x: x.head(1) for k in other_cols},
                    **{k: lambda x: np.concatenate(x.to_list()) for k in array_cols},
                }
            )
            .reset_index(drop=True)
        )
        del df["epoch"]
        param_cols = df.columns[: df.columns.to_list().index("param_id")]
        df = df.sort_values([*param_cols, "rep_id"]).reset_index(drop=True)
        return df
