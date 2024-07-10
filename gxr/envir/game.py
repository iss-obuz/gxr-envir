from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND

from .config import Config
from .dynamics import EnvirDynamics, EnvirDynamicsResults
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

    def __init__(self, model: EnvirModel, *, dt: float = 0.1) -> None:
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
    def reward(self) -> FloatND:
        return self.model.reward

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

    def make_record(self, sol: EnvirDynamicsResults) -> Mapping[str, Any]:
        """Make record of a simulation run."""
        return {
            "n_agents": self.n_agents,
            "horizon": self.horizon,
            "alpha": self.alpha,
            "delay": self.delay,
            "dt": sol.T[-1] - sol.T[0],
            "E0": sol.E[0],
            "P0": sol.P[..., 0],
            "H0": sol.H[..., 0],
            "dR": sol.R[-1] - sol.R[0],
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        n_agents: int | None = None,
        horizon: float | None = None,
        alpha: float | None = None,
        delay: float | None = None,
        noise: float | None = None,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Self:
        """Initialize game instance from ``config``.

        Parameters
        ----------
        config
            Config mapping. Use default values when ``None``.
            Supplied values overrides defaults.
        n_agents
            Utility argument for overriding the ``n_agents`` field in the config.
        horizon
            Utility argument for overriding the ``foresight`` field in the config.
        alpha
            Utility argument for overriding the ``alpha`` field in the config.
        delay
            Utility argument for overriding the ``delay`` field in the config.
        noise
            Utility argument for overriding the ``noise`` field in the config.
        **kwargs
            Passed to :meth:`__init__`.
        """
        config = Config(config or {})

        if n_agents is not None:
            config["model"]["n_agents"] = n_agents

        model = EnvirModel(**config["model"])
        game = cls(model, **kwargs)

        if horizon is not None:
            game.horizon = horizon
        if alpha is not None:
            game.alpha = alpha
        if delay is not None:
            game.delay = delay
        if noise is not None:
            game.noise = noise

        if isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)
        if random_state is not None:
            game.model.behavior.rng = random_state

        return game
