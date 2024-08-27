from collections.abc import Mapping
from typing import Any, Literal, Self

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
        dt: float = 0.002,
    ) -> None:
        self.model = model
        self.dynamics = EnvirDynamics(self.model)
        self.dH = np.zeros_like(self.model.P)
        self.Ehat = self.model.E
        self.dt = dt * self.model.envir.T_epsilon
        self.params = EnvirGameParams(self)

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

    def step(self, H: FloatND | None = None) -> None:
        """Run one simulation step."""
        if H is None:
            H = np.zeros_like(self.H)
        self.H = H
        dE = self.model.get_dE(self.E, self.H)
        dP = self.model.get_dP(self.E, self.H)
        dEhat = self.model.get_dEhat(self.E, self.Ehat)
        self.dH = self.model.behavior.dH(self.E, self.H, self.P)
        self.model.T += self.dt
        self.model.E += dE * self.dt
        self.model._P += dP * self.dt
        self.Ehat += dEhat * self.dt

    def get_results(self, sol: EnvirDynamicsResults) -> Mapping[str, Any]:
        """Get full simulation results."""
        T = sol.T.astype(np.float32)
        return {
            "epoch": None,
            "epochs": T / self.model.envir.T_epsilon,
            "T": T,
            "E": sol.E.astype(np.float32),
            "H": sol.H.flatten().astype(np.float32),
            "P": sol.P.flatten().astype(np.float32),
            "U": self.utility(sol.P).flatten().astype(np.float32),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        no_behavior: bool = False,
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
        cls_kwargs = {}
        if "dt" in kwargs:
            cls_kwargs["dt"] = kwargs.pop("dt")
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

        config = Config(config, resolve=False, interpolate=True)

        if no_behavior:
            config["model"]["behavior"]["rules"] = {"constant": {"@rules": "constant"}}
            config = Config(config, resolve=True)
            config["model"]["behavior"].rules_map = {
                k: v
                for k, v in config["model"]["behavior"].rules_map.items()
                if k == "constant"
            }
        else:
            config = Config(config, resolve=True)

        model = EnvirModel(**config["model"])
        return cls(model, **cls_kwargs)


class EnvirGameParams:
    """Parameter manager.

    Attributes
    ----------
    game
        Game instance.
    """

    def __init__(
        self,
        game: EnvirGame,
        *,
        max_horizon: float = 5.0,
        max_delay: float = 5.0,
        min_nz: float = 0.01,
        n_steps: int = 10,
    ) -> None:
        self.game = game
        self.max_horizon = max_horizon
        self.max_delay = max_delay
        self.min_nz = min_nz
        self.n_steps = n_steps

    @property
    def horizon(self) -> float:
        return self.game.model.foresight.horizon

    @horizon.setter
    def horizon(self, value: float) -> None:
        self.game.model.foresight.horizon = max(
            self.min_nz, min(value, self.max_horizon)
        )

    @property
    def alpha(self) -> float:
        return self.game.model.behavior.rules_map["foresight"].alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.game.model.behavior.rules_map["foresight"].alpha = max(0, min(1, value))

    @property
    def delay(self) -> float:
        return self.game.model.behavior.delay

    @delay.setter
    def delay(self, value: float) -> None:
        self.game.model.behavior.delay = max(self.min_nz, min(value, self.max_delay))

    @property
    def bias(self) -> float:
        return self.game.model.behavior.bias

    @bias.setter
    def bias(self, value: float) -> None:
        self.game.model.behavior.bias = max(0, min(1, value))

    @property
    def horizon_step(self) -> float:
        return (self.max_horizon - self.min_nz) / self.n_steps

    @property
    def alpha_step(self) -> float:
        return 1 / self.n_steps

    @property
    def delay_step(self) -> float:
        return (self.max_delay - self.min_nz) / self.n_steps

    @property
    def bias_step(self) -> float:
        return 1 / self.n_steps

    def change(
        self,
        *,
        horizon: Literal[-1, 0, 1] = 0,
        alpha: Literal[-1, 0, 1] = 0,
        delay: Literal[-1, 0, 1] = 0,
        bias: Literal[-1, 0, 1] = 0,
    ) -> float:
        """Increase or decrease parameters.

        Returns
        -------
        n_changes
            Number of standardized changes for using in regularization.
        """
        if horizon > 0:
            self.horizon += self.horizon_step
        elif horizon < 0:
            self.horizon -= self.horizon_step
        if alpha > 0:
            self.alpha += self.alpha_step
        elif alpha < 0:
            self.alpha -= self.alpha_step
        if delay > 0:
            self.delay += self.delay_step
        elif delay < 0:
            self.delay -= self.delay_step
        if bias > 0:
            self.bias += self.bias_step
        elif bias < 0:
            self.bias -= self.bias_step

        return float(abs(horizon) + abs(alpha) + abs(delay) + abs(bias))
