from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from gxr.dotpath import dotset
from gxr.envir.config import registry
from gxr.typing import FloatND

from .config import Config
from .dynamics import EnvirDynamics, EnvirDynamicsResults
from .functions.utility import UtilityFunction
from .model import EnvirModel


@registry.envir.modules.register("game")
class EnvirGame:
    r"""Environmental game.

    Attributes
    ----------
    model
        Environment model instance.
    dt
        Time step length.
    dH
        Current perceived optimal harvesting rate change.
    Ehat
        Current perceived environment state.
    """

    def __init__(
        self,
        model: EnvirModel,
        *,
        dt: float = 0.1,
    ) -> None:
        self.model = model
        self.dynamics = EnvirDynamics(self.model)
        self.dH = np.zeros_like(self.model.P)
        self.Ehat = self.model.E
        self.dt = dt

    @property
    def n_agents(self) -> int:
        return self.model.n_agents

    @property
    def E(self) -> FloatND:
        return self.model.E

    @E.setter
    def E(self, value: FloatND) -> None:
        self.model.E = value

    @property
    def P(self) -> FloatND:
        return self.model.P

    @P.setter
    def P(self, value: FloatND) -> None:
        self.model.P = value

    @property
    def H(self) -> FloatND:
        return self.model.H

    @H.setter
    def H(self, value: FloatND) -> None:
        self.model.H = value

    @property
    def U(self) -> FloatND:
        return self.model.U

    @property
    def horizon(self) -> float:
        return self.model.foresight.horizon

    @horizon.setter
    def horizon(self, value: float) -> None:
        self.model.foresight.horizon = value

    @property
    def alpha(self) -> float:
        return self.model.behavior.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.model.behavior.alpha = value

    @property
    def noise(self) -> float:
        return self.model.behavior.noise

    @noise.setter
    def noise(self, value: float) -> None:
        self.model.behavior.noise = value

    @property
    def delay(self) -> float:
        return self.model.behavior.delay

    @delay.setter
    def delay(self, value: float) -> None:
        self.model.behavior.delay = value

    @property
    def cost(self) -> float:
        return self.model.profits.cost

    @cost.setter
    def cost(self, value: float) -> None:
        self.model.profits.cost = max(0, value)

    @property
    def utility(self) -> UtilityFunction:
        return self.model.utility

    # Methods --------------------------------------------------------------------------

    def step(self, H: FloatND | None) -> None:
        """Run one simulation step."""
        self.H = H
        dE = self.model.get_dE(self.E, self.H)
        dP = self.model.get_dP(self.E, self.H)
        dEhat = self.model.get_dEhat(self.E, self.Ehat)
        self.dH = self.model.behavior.dH(self.E, self.H)
        self.model.E += dE * self.dt
        self.model._P += dP * self.dt
        self.Ehat += dEhat * self.dt

    def get_results(self, sol: EnvirDynamicsResults) -> Mapping[str, Any]:
        """Get full simulation results."""
        T = sol.T.astype(np.float32)
        return {
            "epochs": T / self.model.envir.K,
            "T": T,
            "H": sol.H.flatten().astype(np.float32),
            "P": sol.P.flatten().astype(np.float32),
            "U": self.utility(sol.P).flatten().astype(np.float32),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        *,
        dt: float = 0.1,
        **kwargs: Any,
    ) -> Self:
        """Initialize game instance from ``config``.

        Parameters
        ----------
        override
            Config values to override specified as a flat mapping
            using dotpath notation.
        **kwargs
            Can be used to define `"params.<name>"` keys in the
            ``overrides`` dictionary for convenience.
        """
        if not config:
            config = Config(resolve=False, interpolate=False)
        overrides = overrides or {}

        allowed_params = list(Config()["params"])
        for k, v in kwargs.items():
            if k not in allowed_params:
                errmsg = f"'{k}' is not a valid configuration parameter"
                raise ValueError(errmsg)
            overrides[f"params.{k}"] = v

        for k, v in overrides.items():
            dotset(config, k, v, item=True)

        config = Config(config, resolve=True, interpolate=True)
        model = EnvirModel(**config["model"])
        return cls(model, dt=dt)
