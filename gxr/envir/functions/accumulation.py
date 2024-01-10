from typing import Any
from warnings import catch_warnings, filterwarnings
import numpy as np
from .abc import StateFunction
from .envir import Envir
from ...typing import FloatND


class Accumulation(StateFunction):
    """Accumulation function.

    Attributes
    ----------
    envir
        Environment function.
    """
    def __init__(
        self,
        envir: Envir,
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.envir = envir

    def __call__(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Accumulated environment state.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``h``.
        E0, h
            Initial state and harvesting rate.
            Must be mutually broadcastable.
        """
        T = np.array(t)
        E0, h = np.broadcast_arrays(E0, h)
        out = np.broadcast(T, E0, h)
        rh, Kh = self.envir.adjust_params(*self.envir.get_params(), h)
        with catch_warnings():
            filterwarnings("ignore", "overflow|invalid|divide")
            D = E0/Kh
            rh = rh.reshape(D.shape)
            A = np.atleast_1d((Kh/rh*np.log(np.abs((1-D)*np.exp(-rh*T) + D)) + Kh*T))
        mask = np.isclose(rh, 0, **self.limtol)
        if mask.any():
            r, K = self.envir.get_params()
            mask = np.broadcast_to(mask, A.shape)
            limit = np.broadcast_to(K/r*np.log(np.abs(1 + E0/K*r*T)), A.shape)
            A[mask] = limit[mask]
        return A.reshape(out.shape)

    def tpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND
    ) -> float | FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``h``.
        E0, h
            Initial state and harvesting rate.
            Must be mutually broadcastable.
        """
        return self.envir(t, E0, h)

    def hpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND
    ) -> float | FloatND:
        """Partial derivative with respect to harvesting rate.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``h``.
        E0, h
            Initial state and harvesting rate.
            Must be mutually broadcastable.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        T = self.make_grid(0, t)
        dE = self.envir.hpartial(T, E0, h)
        dA = np.trapz(dE.T, x=T.T, axis=-1).T
        return dA
