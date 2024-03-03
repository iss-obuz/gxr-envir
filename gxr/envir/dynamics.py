from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm as TqdmPBar
from tqdm.autonotebook import tqdm

from gxr.typing import FloatND

from .behavior import Behavior
from .model import EnvirModel


class EnvirDynamicsResults:
    """Environmental dynamics simulation results.

    Attributes
    ----------
    ode
        ODE system solution.
    """

    def __init__(self, ode: OdeResult) -> None:
        self.ode = ode

    @property
    def n_agents(self) -> int:
        return (len(self.ode.y) - 1) // 2

    @property
    def T(self) -> FloatND:
        """Time grid of the ODE solution."""
        return np.atleast_1d(self.ode.t)

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

    @property
    def H(self) -> FloatND:
        """Harvesting rates over the time grid."""
        return np.atleast_1d(self.ode.y[-self.n_agents :])

    def get_arrays(self) -> tuple[FloatND, FloatND, FloatND, FloatND]:
        """Get time grid and all results array.

        Returns
        -------
        T
            Tiem grid.
        E
            Environment states.
        P
            Agents' profits.
        H
            Individual harvesting rates.
        """
        return self.T, self.E, self.P, self.H


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
        progress: bool | dict = False,
        tol: float | tuple[float, float] = 1e-2,
        etol: float | tuple[float, float] | None = 1e-3,
        ptol: float | tuple[float, float] | None = None,
        htol: float | tuple[float, float] | None = None,
        **kwds: Any,
    ) -> EnvirDynamicsResults:
        """Run environmental dynamics simulation by solving the ODE system.

        Parameters
        ----------
        t
            Maximum time for which to solve.
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
        **kwds
            Passed to :func:`scipy.integrate.solve_ivp`.
            Argument ``args`` cannot be used.
        """
        disable = not progress
        if not isinstance(progress, dict):
            progress = {}
        pbar = tqdm(
            total=t,
            disable=disable,
            **{
                "desc": "Solving ODE system",
                "delay": 2,
                "bar_format": "{l_bar}{bar}{n:.2f}/{total_fmt} "
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
        t_span = (0, t)
        sol = solve_ivp(
            self.ode,
            t_span,
            self.get_y0(),
            args=(pbar,),
            **{"method": "RK23", "rtol": rtol, "atol": atol, **kwds},
        )
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
        return np.array([E, E, *self.model.P.copy(), *self.model.H.copy()])

    def get_vicious_bounds(self, sol: EnvirDynamicsResults) -> FloatND:
        """Get bounds of the vicious cycle region."""
        return self.model.get_vicious_bounds(sol.H.sum(axis=0))

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
