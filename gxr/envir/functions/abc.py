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
        t
            Time.
        E0, h
            Initial state and harvesting rate.
            Must be mutually broadcastable.
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
        t, h
            Time and harvesting rates.
            Must be of the same shape.
        E0
            Initial state.
            Must be broadcastable with ``h``.
        """
        t, h = make_arrays(t, h)
        if t.shape != h.shape:
            raise ValueError("'t' and 'h' have to be of the same shape")
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
    def align_with_H(H: FloatND, *Xs: np.ndarray) -> tuple[np.ndarray, ...]:
        if not Xs:
            raise ValueError("no arrays to align")
        H, *Xs = make_arrays(H, *Xs)
        *Xs, _ = np.broadcast_arrays(*Xs, H[0])
        n_dims = Xs[0].ndim - H[0].ndim
        H = expand_dims(H, n_dims, axis=1)
        return (*Xs, H)

    @staticmethod
    def align_agent_arrays(*Xs: np.ndarray) -> tuple[np.ndarray, ...]:
        Xs = (np.moveaxis(np.atleast_1d(x), 0, -1) for x in Xs)
        Xs = np.broadcast_arrays(*Xs)
        Xs = (np.moveaxis(x, -1, 0) for x in Xs)
        return Xs

    def _gradient(self, *args: Any, H: FloatND, **kwds) -> FloatND:
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        n_agents = len(np.atleast_1d(H))
        idx = np.arange(n_agents)
        pXH = np.repeat(np.expand_dims(pXhj, 0), n_agents, axis=0)
        pXH[idx, idx] = pXhi
        return np.concatenate([pXt[:, None, ...], pXH], axis=1)

    def _tderiv(
        self,
        t: float | FloatND,
        H: float | FloatND = 0,
        *args: Any,
        _time_dependent: bool = True,
        **kwds: Any
    ) ->  float | FloatND:
        """Time path derivative."""
        t, H = make_arrays(t, H)
        if t.shape != H[0].shape:
            raise ValueError("'t' and 'H[0]' have to be of the same shape")
        dH  = numderiv(H, t[None, ...], axis=1)
        dh  = dH.sum(axis=0)
        if _time_dependent:
            args = (t, *args)
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        dP = pXt + dH*pXhi + (dh-dH)*pXhj
        return dP
