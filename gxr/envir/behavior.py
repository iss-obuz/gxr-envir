from abc import ABC, abstractmethod
from math import log
from typing import TYPE_CHECKING, Any

import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND

from .functions import Envir, Foresight, Utility
from .functions.utility import UtilityFunction

if TYPE_CHECKING:
    from .model import EnvirModel


@registry.envir.modules.register("behavior")
class Behavior:
    """Agents' behavior definition.

    Attributes
    ----------
    model
        Environmental model instance.
    delay
        Number of characteristic timescales needed for the perceived state
        at the level of carrying capacity update to ``K/10``.
        Must be positive. Lower values indicate less delay.
    bias
        Bias towards higher environment states.
        Must be between 0 and 1.
    eta
        Adaptation rate.
    alpha
        Coordination strength expected by agents.
    noise
        Behavior noise magnitude.
    rules
        Mapping with behavior rules.
    """

    def __init__(
        self,
        model: "EnvirModel | None" = None,
        *,
        delay: float = 1,
        bias: float = 0,
        eta: float = 0.2,
        noise: float = 0.5,
        random_state: int | np.random.Generator | None = None,
        rules: dict[str, "BehaviorRule"],
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        seed
            Seed for random number generator.
        """
        self.model = model
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(np.random.PCG64(random_state))
        )
        self.delay = delay
        self.bias = bias
        self.eta = eta
        self.noise = noise
        self.rules_map = rules
        for k, v in self.rules_map.items():
            v.behavior = self
            self.rules_map[k] = v

    def __getattr__(self, attr: str) -> Any:
        try:
            return self.rules_map[attr]
        except KeyError as exc:
            errmsg = f"'{self.__class__.__name__}' object has not attribute '{attr}'"
            raise AttributeError(errmsg) from exc

    @property
    def n_agents(self) -> int:
        return self.model.n_agents

    @property
    def envir(self) -> Envir:
        return self.model.envir

    @property
    def foresight(self) -> Foresight:
        return self.model.foresight

    @property
    def adaptation_rate(self) -> float:
        return self.eta / self.envir.T_epsilon

    @property
    def sigma(self) -> float:
        return self.noise * self.envir.T_epsilon / self.n_agents

    @property
    def rules(self) -> list["BehaviorRule"]:
        return list(self.rules_map.values())

    def dH(self, E: float, H: FloatND, P: FloatND) -> FloatND:
        """Determine change of agents' harvesting rates."""
        denom = 0
        dH = np.zeros(self.n_agents, dtype=float)
        for rule in self.rules:
            denom += rule.weight
            dH += rule.weight * rule.dH(E, H, P)
        dH /= denom
        if self.noise and self.noise > 0:
            dH += self.rng.normal(0, self.sigma, dH.shape)
        return np.clip(dH * self.adaptation_rate, -H, None)

    def Ehat_deriv(self, E: float, Ehat: float) -> float:
        """Get time derivative of the perceived environment state."""
        coef = log(2) / (self.delay * self.envir.T_epsilon)
        K = self.envir.K
        return coef * ((E - Ehat) + self.bias * (K - E))


class BehaviorRule(ABC):
    """Behavior rule for tweaking harvesting rates.

    Attributes
    ----------
    behavior
        Behavior definition instance.
    weight
        Weight of the rule.
    """

    def __init__(
        self,
        behavior: Behavior | None = None,
        *,
        weight: float = 1,
    ) -> None:
        self.behavior = behavior
        self.weight = weight

    def __call__(self, *args: Any, **kwds: Any) -> FloatND:  # noqa
        """Determine change of agents' harvesting rates."""
        return self.dH(*args, **kwds)

    @property
    def model(self) -> "EnvirModel | None":
        return self.behavior.model

    @property
    def envir(self) -> Envir:
        return self.model.envir

    @property
    def n_agents(self) -> int:
        return self.model.n_agents

    @abstractmethod
    def dH(self, E: float, H: FloatND, P: FloatND) -> FloatND:
        """Determine change of agents' harvesting rates."""


@registry.envir.rules.register("constant")
class ConstantRule(BehaviorRule):
    """Constant behavior rule."""

    def dH(self, E: float, H: FloatND, P: FloatND) -> FloatND:  # noqa
        """Determine change of agents' harvesting rates."""
        return np.zeros_like(H)


@registry.envir.rules.register("foresight")
class ForesightRule(BehaviorRule):
    """Foresight behavior rule.

    Attributes
    ----------
    alpha
        Coordination strength expected by agents.
    """

    def __init__(
        self,
        behavior: Behavior | None = None,
        alpha: float = 0.0,
        **kwds: Any,
    ) -> None:
        super().__init__(behavior, **kwds)
        self.alpha = alpha

    @property
    def foresight(self) -> Foresight:
        return self.model.foresight

    @property
    def utilfunc(self) -> UtilityFunction:
        return self.model.utility

    @property
    def utility(self) -> Utility:
        return Utility(self.foresight)

    def dH(self, E: float, H: FloatND, P: FloatND) -> FloatND:  # noqa
        """Determine change of agents' harvesting rates."""
        dUi, dUj = self.utility.hpartial(E, H)
        weight = self.alpha * (self.n_agents - 1)
        F = (dUi + weight * dUj) / (1 + weight)
        return F / self.envir.K
