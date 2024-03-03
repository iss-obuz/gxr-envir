# ruff: noqa: RET504
from typing import Any
from warnings import catch_warnings, filterwarnings

import numpy as np

from gxr.typing import FloatND

from .abc import StateFunction
from .envir import Envir


class Accumulation(StateFunction):
    """Accumulation function.

    Attributes
    ----------
    envir
        Environment function.
    """

    def __init__(self, envir: Envir, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.envir = envir

    def __call__(self, t: FloatND, E0: FloatND, h: FloatND = 0) -> FloatND:
        """Accumulated environment state.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        rh, Kh = self.envir.adjust_params(*self.envir.get_params(), h)
        with catch_warnings():
            filterwarnings("ignore", "overflow|invalid|divide")
            D = E0 / Kh
            rh = rh.reshape(D.shape)
            A = np.atleast_1d(
                Kh / rh * np.log(np.abs((1 - D) * np.exp(-rh * t) + D)) + Kh * t
            )
        mask = np.isclose(rh, 0, **self.limtol)
        if mask.any():
            r, K = self.envir.get_params()
            mask = np.broadcast_to(mask, A.shape)
            limit = np.broadcast_to(K / r * np.log(np.abs(1 + E0 / K * r * t)), A.shape)
            A[mask] = limit[mask]
        return A.reshape(t.shape)

    def tpartial(self, t: FloatND, E0: FloatND, h: FloatND) -> FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        return self.envir(t, E0, h)

    def hpartial(self, t: FloatND, E0: FloatND, h: FloatND) -> FloatND:
        """Partial derivative with respect to harvesting rate.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable in ``t, E0, h`` order.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        T = self.make_grid(0, t)
        dE = self.envir.hpartial(T, E0, h)
        dA = np.trapz(dE, x=T, axis=0)
        return dA
