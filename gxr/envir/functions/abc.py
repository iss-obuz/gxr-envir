from typing import Any, Self, Mapping
from abc import ABC, abstractmethod
from math import ceil
import numpy as np
from ...typing import FloatND
from ...utils.misc import obj_from_dict
from ...utils.array import make_arrays, numderiv, expand_dims


class Function(ABC):
    """Mathematical function class."""
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Function computations."""

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        """Construct from a mapping."""
        dct = {"@factory": cls.__name__, **dct}
        return obj_from_dict(dct, package=cls.__module__)


class DifferentiableFunction(Function):
    """Differentiable function class."""
    @abstractmethod
    def deriv(self, *args: Any, **kwds: Any) -> Any:
        """Function derivative."""


class ModelFunction(Function):
    """Generic model function.

    Parameters
    ----------
    numint_step_size
        Target step size in numerical integration used by default.
    numint_min_steps
        Minimum number of steps used for numerical integration.
    numint_max_steps
        Maximum number of steps used for numerical integration.
    """
    limtol = {"rtol": 1e-6, "atol": 1e-6}

    def __init__(
        self,
        *,
        numint_step_size: float = .5,
        numint_min_steps: int = 20,
        numint_max_steps: int = 100
    ) -> None:
        self.numint_step_size = numint_step_size
        self.numint_min_steps = numint_min_steps
        self.numint_max_steps = numint_max_steps

    @abstractmethod
    def tpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Partial derivative with respect to time."""

    @abstractmethod
    def hpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Partial derivative with respect to harvesting rate."""

    @abstractmethod
    def gradient(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Gradient."""

    @abstractmethod
    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) ->  float | FloatND:
        """Time path derivative."""

    def make_grid(self, start: FloatND, stop: FloatND, **kwds: Any) -> FloatND:
        """Get time grid for numerical integration over the future."""
        span = abs(np.max(stop) - np.min(start))
        n_steps = ceil(span/self.numint_step_size)
        n_steps = max(min(n_steps, self.numint_max_steps), self.numint_min_steps)
        return np.linspace(start, stop, **{"num": n_steps, **kwds})


class StateFunction(ModelFunction):
    """Generic global state function."""

    def gradient(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Gradient.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        return np.stack([self.tpartial(t, E0, h), self.hpartial(t, E0, h)], axis=0)

    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) ->  float | FloatND:
        """Time path derivative.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        dh = numderiv(h, t, axis=0)
        dX = self.tpartial(t, E0, h) + dh*self.hpartial(t, E0, h)
        return dX


class AgentsFunction(ModelFunction):
    """Generic agents state function."""

    @abstractmethod
    def hpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: FloatND
    ) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect agents' own harvesting rates
        and harvesting rates of another agent.
        """

    @abstractmethod
    def gradient(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND = 0
    ) -> float | FloatND:
        """Gradient."""

    @abstractmethod
    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND = 0
    ) -> float | FloatND:
        """Time path derivative."""

    # Internals ------------------------------------------------------------------------

    @staticmethod
    def make_h(H: FloatND) -> FloatND:
        return np.broadcast_to(H.sum(axis=-1)[..., None], H.shape)

    def _gradient(self, *args: Any, H: FloatND, **kwds) -> FloatND:
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        n_agents = np.atleast_1d(H).shape[-1]
        idx = np.arange(n_agents)
        pXH = np.repeat(pXhj[..., None, :], n_agents, axis=-2)
        pXH[..., idx, idx] = pXhi
        gX = np.concatenate([pXt[..., None, :], pXH], axis=-2)
        return gX

    def _tderiv(
        self,
        t: float | FloatND,
        H: float | FloatND = 0.0,
        *args: Any,
        _time_dependent: bool = True,
        **kwds: Any
    ) ->  float | FloatND:
        """Time path derivative."""
        t, H = make_arrays(t, H)
        if t.shape != H[..., 0].shape:
            raise ValueError("'t' and 'H[..., 0]' have to be of the same shape")
        t = t[..., None]
        dH  = numderiv(H, t, axis=0)
        dh  = dH.sum(axis=-1)
        if _time_dependent:
            args = (t, *args)
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        dP = pXt + dH*pXhi + (dh[..., None]-dH)*pXhj
        return dP
