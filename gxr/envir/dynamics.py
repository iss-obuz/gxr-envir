from typing import Any, Optional
from warnings import catch_warnings, filterwarnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm as TqdmPBar
from tqdm.autonotebook import tqdm
from .game import EnvirGame
from .behavior import Behavior
from ..typing import Float1D, Float2D


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
    def T(self) -> Float1D:
        """Time grid of the ODE solution."""
        return self.ode.t

    @property
    def E(self) -> Float1D:
        """Environment states over the time grid."""
        return self.ode.y[0]

    @property
    def Ehat(self) -> Float1D:
        """Perceived states of the environment."""
        return self.ode.y[1]

    @property
    def P(self) -> Float2D:
        """Agents' profits over the time grid."""
        return self.ode.y[2:2+self.n_agents]

    @property
    def H(self) -> Float2D:
        """Harvesting rates over the time grid."""
        return self.ode.y[-self.n_agents:]

    def get_arrays(self) -> tuple[Float1D, Float1D, Float2D, Float2D]:
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
    game
        Environmental game instance.
    behavior
        Agents' behavior definition instance.
        Should be passed as a dictionary defining behavior parameters.
    """
    def __init__(
        self,
        game: EnvirGame,
        behavior: Optional[dict] = None
    ) -> None:
        self.game = game
        self.behavior = Behavior(game, **(behavior or {}))

    @property
    def n_agents(self) -> int:
        return self.game.n_agents

    def run(
        self,
        t: float,
        *,
        progress: bool | dict = False,
        tol: float | tuple[float, float] = 1e-2,
        etol: Optional[float | tuple[float, float]] = 1e-3,
        ptol: Optional[float | tuple[float, float]] = None,
        htol: Optional[float | tuple[float, float]] = None,
        **kwds: Any
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
        pbar = tqdm(total=t, disable=disable, **{
            "desc": "Solving ODE system",
            "delay": 2,
            "bar_format": "{l_bar}{bar}{n:.2f}/{total_fmt} "
                "[{elapsed}, {rate_fmt}{postfix}]",
            **progress
        })
        tol = self._get_tols(tol)
        ertol, eatol = self._get_tols(etol, tol)
        prtol, patol = self._get_tols(ptol, tol)
        hrtol, hatol = self._get_tols(htol, tol)
        rtol = np.concatenate([
            [ertol]*2,
            np.full(self.n_agents, prtol),
            np.full(self.n_agents, hrtol)
        ])
        atol = np.concatenate([
            [eatol]*2,
            np.full(self.n_agents, patol),
            np.full(self.n_agents, hatol)
        ])
        t_span = (0, t)
        sol = solve_ivp(self.ode, t_span, self.get_y0(), args=(pbar,), **{
            "method": "RK23",
            "rtol": rtol,
            "atol": atol,
            **kwds
        })
        E = sol.y[0, -1]
        P = sol.y[1:self.n_agents+1, -1]
        H = sol.y[-self.n_agents:, -1]
        self.game.E = E
        self.game.P = P
        self.game.H = H
        pbar.close()
        return EnvirDynamicsResults(sol)

    def ode(self, t: float, y: Float1D, pbar: Optional[TqdmPBar] = None) -> Float1D:
        """Dynamics represented as an ODE system.

        The logic and signature of the method is compatible
        with the interface of the :func:`scipy.integrate.solve_ivp`.
        """
        E    = y[0]
        Ehat = y[1]
        P    = y[2:2+self.n_agents]
        H    = y[-self.n_agents:]
        dX   = np.array([
            self.game.envir.deriv(E, H.sum()),
            self.behavior.Ehat_deriv(E, Ehat),
            *self.game.profits.deriv(E, H),
            *self.behavior.dH(Ehat, H, P)
        ])
        if pbar is not None and (deltat := t - pbar.n) > 0:
            pbar.update(deltat)
        return dX

    def get_y0(self) -> Float1D:
        """Get initial state for ODE."""
        E = self.game.E
        return np.array([E, E, *self.game.P.copy(), *self.game.H.copy()])

    def get_vicious_bounds(self, sol: EnvirDynamicsResults) -> Float2D:
        """Get bounds of the vicious cycle region."""
        h = sol.H.sum(axis=0)
        with catch_warnings():
            filterwarnings("ignore", "divide")
            B = np.column_stack([
                self.game.envir.K*(1 - h/self.game.envir.r),
                self.n_agents*(self.game.profits.cost + self.game.profits.sustenance/h)
            ])
        B[B[:, 0] > B[:, 1]] = np.nan
        return B

    # Iternals -------------------------------------------------------------------------

    def _get_tols(
        self,
        tol: None | float | tuple[float, float],
        defaults: Optional[float | tuple[float, float]] = None
    ) -> tuple[float, float]:
        if tol is None:
            if defaults is not None:
                return self._get_tols(defaults)
            raise ValueError("no tolerance levels an no default passed")
        if isinstance(tol, tuple):
            rtol, atol = tol
        else:
            rtol = atol = tol
        return rtol, atol
