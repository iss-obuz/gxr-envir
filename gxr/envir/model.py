from warnings import catch_warnings, filterwarnings

import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND
from gxr.utils.array import make_arrays

from .behavior import Behavior
from .functions import Envir, Foresight, Profits
from .functions.utility import UtilityFunction


@registry.envir.modules.register("model")
class EnvirModel:
    """Environment model.

    Attributes
    ----------
    n_agents
        Number of agents.
    foresight
        Agents' foresight function.
        Profits and environment functions are contained within.
    utility
        Utility function.
    behavior
        Behavior rules.
    E
        Environment state.
    P
        Agents' profits.
    H
        Agents' harvesting rates.
    scale_capacity
        Should carrying capactity be rescale proportionally to the number of agents.
    """

    def __init__(
        self,
        *,
        n_agents: int,
        foresight: Foresight,
        utility: UtilityFunction,
        behavior: Behavior,
        E0: float = 1.0,
        P0: FloatND | None = None,
        H0: FloatND | None = None,
        scale_capacity: bool = True,
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        scale_capacity
            Should carrying capacity and cost rates be rescaled by ``n_agents``.
        """
        if E0 <= 0:
            errmsg = "'E0' has to be positive"
            raise ValueError(errmsg)
        if P0 is None:
            P0 = np.zeros(n_agents)  # type: ignore
        if H0 is None:
            H0 = np.zeros(n_agents)  # type: ignore
        if P0.shape != H0.shape != (n_agents,):
            errmsg = "'P0' and 'H0' have to be 1D with length 'n_agents'"
            raise ValueError(errmsg)

        P0, H0 = make_arrays(P0, H0)
        if P0.ndim != H0.ndim != 1:
            errmsg = "'P0' and 'H0' have to be 1D"
            raise ValueError(errmsg)
        if P0.shape != H0.shape:
            errmsg = "'P0' and 'H0' have to be of the same shape"
            raise ValueError(errmsg)

        self.E = E0
        self._P = P0
        self._H = H0
        self.foresight = foresight
        self.utility = utility
        self.behavior = behavior
        self.behavior.model = self

        if scale_capacity:
            self.envir.rescale_capacity(n_agents)
            self.profits.rescale_cost_rates(n_agents)

        self.E *= self.envir.K

    @property
    def n_agents(self) -> int:
        return len(self.P)

    @property
    def P(self) -> FloatND:
        """Current agents' profit values."""
        return self._P.copy()

    @P.setter
    def P(self, value: FloatND) -> None:
        self._P = self._validate_agent_array(value).copy()

    @property
    def H(self) -> FloatND:
        """Current agents' harvesting rates."""
        return self._H.copy()

    @H.setter
    def H(self, value: FloatND) -> None:
        self._H = self._validate_agent_array(value).copy()

    @property
    def U(self) -> FloatND:
        """Current agents' utility values."""
        return self.utility(self.P)

    @property
    def envir(self) -> Envir:
        """Environment state function."""
        return self.profits.envir

    @property
    def profits(self) -> Profits:
        """Agents' profits function."""
        return self.foresight.profits

    @property
    def reward(self) -> FloatND:
        return np.clip(self.P, 0, np.inf).prod(0) ** (1 / self.n_agents)

    # Methods --------------------------------------------------------------------------

    def get_dE(self, E: FloatND, H: FloatND) -> FloatND:
        """Get environment state derivative."""
        return self.envir.deriv(E, H.sum())

    def get_dEhat(self, E: FloatND, Ehat: FloatND) -> FloatND:
        """Get derivative of the perceived state of the environment."""
        return self.behavior.Ehat_deriv(E, Ehat)

    def get_dP(self, E: FloatND, H: FloatND) -> FloatND:
        """Get profits derivative."""
        return self.profits.deriv(E, H)

    def get_dH(self, Ehat: FloatND, H: FloatND, P: FloatND) -> FloatND:
        """Get derivatives of individual harvesting rates."""
        return self.behavior.dH(Ehat, H, P)

    def deriv(self, E: FloatND, Ehat: FloatND, H: FloatND, P: FloatND) -> FloatND:
        """Get model derivative."""
        return np.array(
            [
                self.get_dE(E, H),
                self.get_dEhat(E, Ehat),
                *np.atleast_1d(self.get_dP(E, H)),
                *np.atleast_1d(self.get_dH(Ehat, H, P)),
            ]
        )

    def get_vicious_bounds(self, h: FloatND) -> FloatND:
        """Get vicious bounds for a sequence of overall harvesting rates.

        Parameters
        ----------
        h
            1D array with overall harvesting rates.
        """
        with catch_warnings():
            filterwarnings("ignore", "divide")
            B = np.column_stack(
                [
                    self.envir.K * (1 - h / self.envir.r),
                    self.n_agents * (self.profits.cost + self.profits.sustenance / h),
                ]
            )
        B[B[:, 0] > B[:, 1]] = np.nan  # type: ignore
        return B

    # Internals ------------------------------------------------------------------------

    def _validate_agent_array(self, X: FloatND) -> FloatND:
        X = np.array(X)
        if X.shape != (self.n_agents,):
            errmsg = f"agent arrays have to be 1D with length {self.n_agents}"
            raise ValueError(errmsg)
        return X
