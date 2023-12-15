from typing import Optional, Any, Union, Self, Iterable, Mapping
from abc import ABC, abstractmethod
import numpy as np
from .game import EnvirGame
from .functions import Envir, Foresight
from ..typing import Float1D
from ..utils.misc import obj_from_dict


class Behavior:
    """Agents' behavior definition.

    Attributes
    ----------
    game
        Environmental game instance.
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
        game: EnvirGame,
        *,
        eta: float = .2,
        noise: float = 1.0,
        seed: Optional[int] = None,
        rules: Iterable[Union[dict, "BehaviorRule"]]
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        seed
            Seed for random number generator.
        """
        self.game = game
        self.rng = np.random.default_rng(seed)
        self.eta = eta
        self.noise = noise
        self.rules = [
            BehaviorRule.from_dict({"behavior": self, **rule})
            if isinstance(rule, Mapping) else rule
            for rule in rules
        ]

    @property
    def n_agents(self) -> int:
        return self.game.n_agents

    @property
    def envir(self) -> Envir:
        return self.game.envir

    @property
    def foresight(self) -> Foresight:
        return self.game.foresight

    @property
    def adaptation_rate(self) -> float:
        return self.eta * self.envir.r/2

    @property
    def sigma(self) -> float:
        return self.noise * self.envir.r/(10*self.n_agents**.5)

    def dH(self, E: float, H: Float1D, P: Float1D) -> Float1D:
        """Determine change of agents' harvesting rates."""
        denom = 0
        dH = np.zeros(self.n_agents, dtype=float)
        for rule in self.rules:
            denom += rule.weight
            dH += rule.weight*rule.dH(E, H, P)
        dH /= denom
        if self.noise and self.noise > 0:
            dH += self.rng.normal(0, self.sigma, dH.shape)
        return np.clip(dH*self.adaptation_rate, -H, None)


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
        behavior: Behavior,
        *,
        weight: float = 1
    ) -> None:
        self.behavior = behavior
        self.weight = weight

    @property
    def game(self) -> EnvirGame:
        return self.behavior.game

    @property
    def envir(self) -> Envir:
        return self.game.envir

    @property
    def n_agents(self) -> int:
        return self.game.n_agents

    @abstractmethod
    def dH(self, E: float, H: Float1D, P: Float1D) -> Float1D:
        """Determine change of agents' harvesting rates."""

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Construct from a dictionary."""
        return obj_from_dict(dct, factory_key="@rule", package=cls.__module__)


class ForesightRule(BehaviorRule):
    """Foresight behavior rule.

    Attributes
    ----------
    alpha
        Coordination strength expected by agents.
    """
    def __init__(self, behavior: Behavior, alpha: float = .0, **kwds: Any) -> None:
        super().__init__(behavior, **kwds)
        self.alpha = alpha

    def dH(self, E: float, H: Float1D, P: Float1D) -> Float1D:
        """Determine change of agents' harvesting rates."""
        dUi, dUj = self.game.utility.hpartial(E, H)
        weight = self.alpha*(self.n_agents-1)
        F = (dUi + weight*dUj) / (1 + weight)
        return F / self.envir.K
