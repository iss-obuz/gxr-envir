from typing import Optional, Any, Union, Sequence, Self, Iterable, Mapping
from math import log
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
    delay
        Number of characteristic timescales needed for the perceived state
        at the level of carrying capacity update to ``K/10``.
        Must be positive. Lower values indicate less delay.
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
        delay: float = 1,
        eta: float = .2,
        noise: float = .5,
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
        self.delay = delay
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
        return self.noise * self.envir.r/(self.n_agents**.5)

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
        if self.bias is not None:
            dH *= self.bias
        return np.clip(dH*self.adaptation_rate, -H, None)

    def Ehat_deriv(self, E: float, Ehat: float) -> float:
        """Get time derivative of the perceived environment state."""
        coef = log(10) / (self.delay * self.envir.T_epsilon)
        return coef * (E - Ehat)


class BehaviorRule(ABC):
    """Behavior rule for tweaking harvesting rates.

    Attributes
    ----------
    behavior
        Behavior definition instance.
    weight
        Weight of the rule.
    bias
        Magnitude of behavior bias passed as a float scalar
        specifying the standard deviation in terms of the multiple
        of the carrying capacity for the zero-center normal distribution.
        Alternatively, a 1D array with per-agent bias values.
    """
    def __init__(
        self,
        behavior: Behavior,
        *,
        weight: float = 1,
        bias: float | None = None
    ) -> None:
        self.behavior = behavior
        self.weight = weight
        self.bias = (
            np.array(bias) if isinstance(bias, Sequence)
            else self.behavior.rng.normal(0, bias*self.envir.K, self.n_agents)
        )

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
        if self.bias is not None:
            E = E + self.bias
        dUi, dUj = self.game.utility.hpartial(E, H)
        weight = self.alpha*(self.n_agents-1)
        F = (dUi + weight*dUj) / (1 + weight)
        return F / self.envir.K
