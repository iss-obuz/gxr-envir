import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND

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
        self.dH = np.zeros_like(self.model.P)
        self.Ehat = self.model.E
        self.dt = dt

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
