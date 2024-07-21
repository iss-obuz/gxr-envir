# ruff: noqa: S311
from collections.abc import Iterable, Iterator
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import tqdm.auto
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm as TqdmPBar

from gxr.typing import FloatND

from .behavior import Behavior
from .model import EnvirModel
from .plotting import DynamicsPlotter


class EnvirDynamicsResults:
    """Environmental dynamics simulation results.

    Attributes
    ----------
    ode
        ODE system solution.
    """

    @dataclass
    class Arrays(Iterable[FloatND]):
        T: FloatND
        E: FloatND
        P: FloatND
        H: FloatND

        def __iter__(self) -> Iterator[FloatND]:
            for field_name in self.__dataclass_fields__:
                yield getattr(self, field_name)

        @property
        def relative(self) -> Self:
            """Get copy with relative times and profits."""
            arr = self.__class__(*(x.copy() for x in self))
            arr.P = arr.P[..., None, 0]
            arr.T -= arr.T[0]
            return arr

        @property
        def n_points(self) -> int:
            """Number of data points."""
            return len(self.T)

        def slice(
            self,
            start: int | None = None,
            stop: int | None = None,
            step: int | None = None,
        ) -> Self:
            idx = slice(start, stop, step)
            return self.__class__(*(x[..., idx] for x in self))

    def __init__(self, ode: OdeResult) -> None:
        self.ode = ode

    def __copy__(self) -> Self:
        return self.__class__(copy(self.ode))

    def __deepcopy__(self) -> Self:
        return self.__class__(deepcopy(self.ode))

    def __getitem__(self, idx: int | slice) -> Self:
        new = self.__copy__()
        new.ode.t = new.ode.t[idx]
        new.ode.y = new.ode.y[..., idx]
        return new

    @property
    def n_agents(self) -> int:
        return (len(self.ode.y) - 1) // 2

    @property
    def n_points(self) -> int:
        return len(self.ode.t)

    @property
    def T(self) -> FloatND:
        """Time grid of the ODE solution."""
        return np.atleast_1d(self.ode.t)

    @T.setter
    def T(self, value: FloatND) -> None:
        self.ode.t = value

    @property
    def E(self) -> FloatND:
        """Environment states over the time grid."""
        return np.atleast_1d(self.ode.y[0])

    @property
    def Ehat(self) -> FloatND:
        """Perceived states of the environment."""
        return np.atleast_1d(self.ode.y[1])

    @property
    def P(self) -> FloatND:
        """Agents' profits over the time grid."""
        return np.atleast_1d(self.ode.y[2 : 2 + self.n_agents])

    @P.setter
    def P(self, value: FloatND) -> None:
        self.ode.y[2 : 2 + self.n_agents] = value

    @property
    def H(self) -> FloatND:
        """Harvesting rates over the time grid."""
        return np.atleast_1d(self.ode.y[-self.n_agents :])

    def get_arrays(self) -> Arrays:
        """Get time grid and all results array.

        Returns
        -------
        T
            Time grid.
        E
            Environment states.
        P
            Agents' profits.
        H
            Individual harvesting rates.
        """
        return self.Arrays(self.T, self.E, self.P, self.H)

    def sample(
        self,
        *,
        relative: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> Self:
        """Get a random sub-history.

        Start and stop time points are sampled uniformly at random.

        Parameters
        ----------
        relative
            Should agents' profits and time be relativized (set to start at zero).
        seed
            Optional random seed for generating sampling points.
        """
        if isinstance(random_state, int | None):
            random_state = np.random.default_rng(random_state)
        size = random_state.integers(2, self.n_points, endpoint=True)
        start = random_state.integers(0, self.n_points - size, endpoint=True)
        subsample = self[start : start + size]
        if relative:
            subsample.T -= subsample.T[0]
            subsample.P -= subsample.P[..., 0, None]
        return subsample


class EnvirDynamics:
    """Dynamical equations of agents and environment.

    Attributes
    ----------
    model
        Environmental model instance.
    """

    def __init__(self, model: EnvirModel) -> None:
        self.model = model

    @property
    def n_agents(self) -> int:
        return self.model.n_agents

    @property
    def behavior(self) -> Behavior:
        return self.model.behavior

    def run(
        self,
        t: float,
        *,
        raw_time: bool = False,
        progress: bool | dict = False,
        tol: float | tuple[float, float] = 1e-2,
        etol: float | tuple[float, float] | None = 1e-2,
        ptol: float | tuple[float, float] | None = None,
        htol: float | tuple[float, float] | None = None,
        n_attempts: int = 5,
        tqdm: type[TqdmPBar] = tqdm.auto.tqdm,
        seed: int | None = None,
        **kwds: Any,
    ) -> EnvirDynamicsResults:
        """Run environmental dynamics simulation by solving the ODE system.

        Parameters
        ----------
        t
            Maximum time for which to solve.
        raw_time
            Use raw time when ``False``.
            Otherwise ``t`` is interpreted as the number of characteristic timescales.
        progress
            Display progress bar.
            Non-empty dicts are interpreted as ``True``
            and passed as additional keyword args to :class:`tqdm.tqdm`.
        tol
            Default tolerance level. If 2-tuple is passed the it interpreted
            as two separate thresholds for ``rtol`` and ``atol``.
        etol
            Tolerance level(s) for environment state.
            Use default values when ``None``.
        ptol
            Tolerance level(s) for profits.
            Use default values when ``None``.
        htol
            Tolerance level(s) for harvesting rates.
            Use default values when ``None``.
        n_attempts
            Number of attempts when the solver does not properly reach the end
            of the integration range.
        tqdm
            :mod:`tqdm` progress bar class to use.
        seed
            Optional seed for controlling the random numbers generator.
        **kwds
            Passed to :func:`scipy.integrate.solve_ivp`.
            Argument ``args`` cannot be used.
        """
        if seed is not None:
            self.behavior.rng = np.random.default_rng(seed)

        if not raw_time:
            t *= self.behavior.envir.T_epsilon
        disable = not progress
        if not isinstance(progress, dict):
            progress = {}
        pbar = tqdm(
            total=t,
            disable=disable,
            **{
                "desc": "Solving ODE system",
                "bar_format": "{l_bar}{bar}{n:.2f}/{total:.2f} "
                "[{elapsed}, {rate_fmt}{postfix}]",
                **progress,
            },
        )
        tol = self._get_tols(tol)
        ertol, eatol = self._get_tols(etol, tol)
        prtol, patol = self._get_tols(ptol, tol)
        hrtol, hatol = self._get_tols(htol, tol)
        rtol = np.concatenate(
            [[ertol] * 2, np.full(self.n_agents, prtol), np.full(self.n_agents, hrtol)]
        )
        atol = np.concatenate(
            [[eatol] * 2, np.full(self.n_agents, patol), np.full(self.n_agents, hatol)]
        )

        # def terminate_numerical(t: float, y: float, *args: Any, **kwds: Any) -> float:  # noqa
        #     return y[0] - 1e-27
        # terminate_numerical.terminal = True  # type: ignore

        # events = kwds.pop("events", [])
        # if isinstance(events, Iterable):
        #     events = list(events)
        # if isinstance(events, list):
        #     events.append(terminate_numerical)

        kwds = {
            "method": "RK23",
            "rtol": rtol,
            "atol": atol,
            # "events": events,
            **kwds,
        }

        t_span = (0, t)
        sol = None
        for _ in range(n_attempts):
            sol = solve_ivp(self.ode, t_span, self.get_y0(), args=(pbar,), **kwds)
            if sol.success:
                break
        if not sol:
            errmsg = f"intergation failed {n_attempts} times; aborting"
            raise RuntimeError(errmsg)

        E = sol.y[0, -1]
        P = sol.y[1 : self.n_agents + 1, -1]
        H = sol.y[-self.n_agents :, -1]
        self.model.E = E
        self.model.P = P
        self.model.H = H
        pbar.close()
        return EnvirDynamicsResults(sol)

    def ode(self, t: float, y: FloatND, pbar: TqdmPBar | None = None) -> FloatND:
        """Dynamics represented as an ODE system.

        The logic and signature of the method is compatible
        with the interface of the :func:`scipy.integrate.solve_ivp`.
        """
        y = np.atleast_1d(y)
        E = y[0]
        Ehat = y[1]
        P = y[2 : 2 + self.n_agents]
        H = y[-self.n_agents :]
        dX = self.model.deriv(E, Ehat, H, P)
        if pbar is not None and (deltat := t - pbar.n) > 0:
            pbar.update(deltat)
        return dX

    def get_y0(self) -> FloatND:
        """Get initial state for ODE."""
        E = self.model.E
        return np.array([E, E, *self.model.P, *self.model.H]).copy()  # type: ignore

    def get_vicious_bounds(self, sol: EnvirDynamicsResults) -> FloatND:
        """Get bounds of the vicious cycle region."""
        return self.model.get_vicious_bounds(sol.H.sum(axis=0))

    def plot(
        self,
        sol: EnvirDynamicsResults,
        *,
        nrows: int = 3,
        ncols: int = 1,
        figsize: tuple[float, float] = (8, 8),
        state_kwargs: dict[str, Any] | None = None,
        harvesting_kwargs: dict[str, Any] | None = None,
        utilities_kwargs: dict[str, Any] | None = None,
    ) -> Figure:
        """Plot solution to the dynamics simulation."""
        plotter = DynamicsPlotter(self, sol)
        fig, axes = plotter.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        state_kwargs = {
            "show_vicious": False,
            "show_perceived": True,
            **(state_kwargs or {}),
        }
        plotter.plot_state(axes[0], **state_kwargs)
        plotter.plot_harvesting(axes[1], **(harvesting_kwargs or {}))
        plotter.plot_utilities(axes[2], **(utilities_kwargs or {}))
        fig.tight_layout()
        return fig

    # Iternals -------------------------------------------------------------------------

    def _get_tols(
        self,
        tol: None | float | tuple[float, float],
        defaults: float | tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        if tol is None:
            if defaults is not None:
                return self._get_tols(defaults)
            errmsg = "no tolerance levels and no defaults passed"
            raise ValueError(errmsg)
        if isinstance(tol, tuple):
            rtol, atol = tol
        else:
            rtol = atol = tol
        return rtol, atol
