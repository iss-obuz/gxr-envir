# ruff: noqa: RET504
from abc import ABC, abstractmethod
from math import ceil
from typing import Any

import numpy as np

from gxr.typing import FloatND
from gxr.utils.array import make_arrays, numderiv


class Function(ABC):
    """Mathematical function class."""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Function computations."""


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
        numint_step_size: float = 0.5,
        numint_min_steps: int = 20,
        numint_max_steps: int = 100,
    ) -> None:
        self.numint_step_size = numint_step_size
        self.numint_min_steps = numint_min_steps
        self.numint_max_steps = numint_max_steps

    @abstractmethod
    def tpartial(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Partial derivative with respect to time."""

    @abstractmethod
    def hpartial(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Partial derivative with respect to harvesting rate."""

    @abstractmethod
    def gradient(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Gradient."""

    @abstractmethod
    def tderiv(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Time path derivative."""

    def make_grid(self, start: FloatND, stop: FloatND, **kwds: Any) -> FloatND:
        """Get time grid for numerical integration over the future."""
        span = abs(np.max(stop) - np.min(start))
        n_steps = ceil(span / self.numint_step_size)
        n_steps = max(min(n_steps, self.numint_max_steps), self.numint_min_steps)
        return np.linspace(start, stop, **{"num": n_steps, **kwds})


class StateFunction(ModelFunction):
    """Generic global state function."""

    def gradient(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Gradient.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        return np.stack([self.tpartial(t, E0, h), self.hpartial(t, E0, h)], axis=0)

    def tderiv(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order
            and ``t`` and ``h`` must be of the same shape.
        """
        t, E0, h = make_arrays(t, E0, h)
        if t.shape != h.shape:
            errmsg = "'t' and 'h' must be of the same shape"
            raise ValueError(errmsg)
        t, E0, h = np.broadcast_arrays(t, E0, h)
        dh = numderiv(h, t, axis=0)
        dX = self.tpartial(t, E0, h) + dh * self.hpartial(t, E0, h)
        return dX


class AgentsFunction(ModelFunction):
    """Generic agents state function."""

    @abstractmethod
    def tpartial(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Partial derivative with respect to time."""

    @abstractmethod
    def hpartial(self, t: FloatND, E0: FloatND, H: FloatND) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect agents' own harvesting rates
        and harvesting rates of another agent.
        """

    @abstractmethod
    def gradient(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Gradient."""

    @abstractmethod
    def tderiv(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Time path derivative."""

    # Internals ------------------------------------------------------------------------

    @staticmethod
    def prepare_data(H: FloatND, *args: FloatND, h: bool = True) -> tuple[FloatND, ...]:
        H = np.atleast_2d(H)
        args = make_arrays(*args)
        is_multiparam_single = H.size == 1 and any(x.size > 1 for x in args)
        *args, H = np.broadcast_arrays(*args, H)  # type: ignore
        args = (*args, H)
        if h:
            args = (*args, np.broadcast_to(H.sum(axis=-1)[..., None], H.shape))
        args = tuple(
            x.squeeze(axis=0) if x.ndim == 2 and x.shape[0] == 1 else x for x in args
        )
        if is_multiparam_single:
            args = (x[..., None] for x in args)  # type: ignore
        return args

    def _gradient(
        self, *args: Any, H: FloatND, _time_dependent: bool = True, **kwds: Any
    ) -> FloatND:
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        n_agents = np.atleast_1d(H).shape[-1]
        pXt = np.atleast_1d(pXt)
        pXhi = np.atleast_1d(pXhi)
        pXhj = np.atleast_1d(pXhj)
        idx = np.arange(n_agents)
        pXH = np.repeat(pXhj[..., None, :], n_agents, axis=-2)
        pXH[..., idx, idx] = pXhi
        gX = np.concatenate([pXt[..., None, :], pXH], axis=-2)
        if not _time_dependent:
            gX = gX[..., 1:, :]
        return gX

    def _tderiv(
        self,
        t: float | FloatND,
        H: float | FloatND = 0.0,
        *args: Any,
        _time_dependent: bool = True,
        **kwds: Any,
    ) -> float | FloatND:
        """Time path derivative."""
        t, H = make_arrays(t, H)
        if t.shape != H[..., 0].shape:
            errmsg = "'t' and 'H[..., 0]' have to be of the same shape"
            raise ValueError(errmsg)
        t = t[..., None]
        dH = numderiv(H, t, axis=0)
        dh = dH.sum(axis=-1)
        if _time_dependent:
            args = (t, *args)
        pXt = self.tpartial(*args, H=H, **kwds)
        pXhi, pXhj = self.hpartial(*args, H=H, **kwds)
        dP = pXt + dH * pXhi + (dh[..., None] - dH) * pXhj
        return dP
