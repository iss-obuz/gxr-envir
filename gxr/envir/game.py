"""Environment model."""
from typing import Any, Optional, Self
import numpy as np
from .functions import Envir, Accumulation, Profits, Foresight, Utility
from .functions.utility import UtilityFunction
from ..typing import PathLike, Float1D
from ..utils.array import make_arrays


class EnvirGame:
    r"""Environmental game.

    Attributes
    ----------
    E
        Environment state.
    P
        Agents' profits.
    H
        Agents' harvesting rates.
    envir
        Environment state function.
    profits
        Agents' profits function.
    foresight
        Foresight function
    utility
        Utility function.
    dt
        Time step length.
    """
    def __init__(
        self,
        E0: float,
        P0: Float1D,
        H0: Float1D,
        utility: Utility,
        *,
        dt: float = .1
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        P0, H0
            Initial profits and individual harvesting rates.
            Must be 1D and of the same length.
        """
        P0, H0 = make_arrays(P0, H0)
        if P0.ndim != H0.ndim != 1:
            raise ValueError("'P0' and 'H0' have to be 1D")
        if P0.shape != H0.shape:
            raise ValueError("'P0' and 'H0' have to be of the same shape")
        self.E = E0
        self._P = P0
        self._H = H0
        self.utility = utility
        self.dt = dt

    # Properties -----------------------------------------------------------------------

    @property
    def envir(self) -> Envir:
        """Environment state function."""
        return self.utility.envir

    @property
    def foresight(self) -> Foresight:
        """Foresight function."""
        return self.utility.foresight

    @property
    def profits(self) -> Profits:
        """Profits function."""
        return self.foresight.profits

    @property
    def n_agents(self) -> int:
        return len(self.P)

    @property
    def P(self) -> Float1D:
        """Current agents' profit values."""
        return self._P.copy()
    @P.setter
    def P(self, value: Float1D) -> None:
        self._P = self._validate_agent_array(value).copy()

    @property
    def H(self) -> Float1D:
        """Current agents' harvesting rates."""
        return self._H.copy()
    @H.setter
    def H(self, value: Float1D) -> None:
        self._H = self._validate_agent_array(value).copy()

    @property
    def U(self) -> Float1D:
        """Current agents' utility values."""
        return self.utility.func(self.P)

    # Methods --------------------------------------------------------------------------

    def step(self, H: Optional[Float1D]) -> None:
        """Run one simulation step."""
        if H is None:
            H = np.zeros(self.n_agents)
        if H.shape != self.P.shape:
            raise ValueError("'H' has to be of the same shape as 'self.P'")
        dE = self.envir.deriv(self.E, H.sum())
        dP = self.profits.deriv(self.E, H)
        self.E += dE*self.dt
        self._P += dP*self.dt

    # Class methods & constructors --------------------------------------------

    @classmethod
    def from_params(
        cls,
        n_agents: int,
        E0: Optional[float] = None,
        P0: Optional[Float1D] = None,
        H0: Optional[Float1D] = None,
        envir: Optional[dict] = None,
        profits: Optional[dict] = None,
        foresight: Optional[dict] = None,
        utility: Optional[dict] = None,
        scale_capacity: bool = True,
        numint_kws: Optional[dict] = None,
        **kwds: Any
    ) -> Self:
        r"""Construct from parameter specifications.

        Parameters
        ----------
        n_agents
            Number of agents.
        E0
            Initial environment state as a fraction of carrying capacity.
        P0, H0
            Initial profits and individual harvesting rates.
            Must agree in size with ``n_agents``. Defaults to zero arrays.
        envir
            Dict with environment parameters,
            ``T``, ``K`` and ``E0`` and ``epsilon``.
        profits
            Dict with profits parameters, ``cost`` and ``sustenance``.
        foresight
            Dict with foresight parameters, ``gamma`` and ``epsilon``.
        utility
            Dict with name of factory function and parameters of the utility function.
        numint_kws
            Numerical integration parameters passed to model functions.
        scale_capacity
            Scale carrying capacity to population size, ``K *= n_agents``.
        **kwds
            Passed to the initialization method.
        """
        if P0 is None:
            P0 = np.zeros(n_agents)
        if H0 is None:
            H0 = np.zeros(n_agents)
        if P0.shape !=  H0.shape != (n_agents,):
            raise ValueError("'P0' and 'H0' have to be 1D with length 'n_agents'")
        numint_kws = numint_kws or {}
        if scale_capacity and "K" in envir:
            envir["K"] *= n_agents
        envir = Envir(**{**numint_kws, **envir})
        profits = profits or {}
        accum = Accumulation(envir, **numint_kws)
        profits = Profits(accum, **profits, **numint_kws)
        profits.rescale_cost_rates(n_agents, envir)
        foresight = foresight or {}
        foresight = Foresight(profits, **foresight, **numint_kws)
        if utility and isinstance(utility, dict):
            utility = {"utility": UtilityFunction.from_dict(utility)}
        elif not utility:
            utility = {}
        utility = Utility(foresight, **utility, **numint_kws)
        if E0 <= 0:
            raise ValueError("'E0' has to be positive")
        E0 *= envir.K
        return cls(E0, P0, H0, utility, **kwds)

    @classmethod
    def from_json(cls, path: PathLike, **kwds: Any) -> Self:
        """Construct from JSON config."""
        return cls(**json.load(path, **kwds))  # type: ignore

    # Internals ------------------------------------------------------------------------

    def _validate_agent_array(self, X: Float1D) -> Float1D:
        X = np.array(X)
        if X.shape != (self.n_agents,):
            raise ValueError(f"agent arrays have to be 1D with length {self.n_agents}")
        return X
