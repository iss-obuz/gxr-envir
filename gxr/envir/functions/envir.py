from typing import Any
from warnings import catch_warnings, filterwarnings
import numpy as np
from .abc import StateFunction
from ...typing import FloatND
from ...utils.array import make_arrays


class Envir(StateFunction):
    r"""Environment function.

    Attributes
    ----------
    r
        Natural growth rate.
    K
        Carrying capacity.
    epsilon
        :math:`\epsilon`-threshold for natural growth rate from timescale.
    """
    _default_epsilon = .01

    def __init__(
        self,
        K: float,
        T: float,
        epsilon: float = _default_epsilon,
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        T
            Characteristic timescale.
        """
        if T <= 0:
            raise ValueError("'T' has to be positive")
        if K <= 0:
            raise ValueError("'K' has to be positive")
        super().__init__(**kwds)
        r = self.get_r_from_timescale(T, epsilon)
        self.r = r
        self.K = K
        self.epsilon = epsilon

    @property
    def T_epsilon(self) -> float:
        """Characteristic timescale."""
        return -2/self.r*np.log(self.epsilon / (1-self.epsilon))

    def __call__(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Environment state.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate.
            Must be jointly broadcastable.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        rh, Kh = self.adjust_params(self.r, self.K, h)
        with catch_warnings():
            filterwarnings("ignore", "overflow|invalid|divide")
            D = (Kh-E0)/E0
            D = D.reshape(rh.shape)
            E = np.atleast_1d(Kh / (1 + D*np.exp(-rh*t)))
        mask = np.isclose(rh, 0, **self.limtol)
        if mask.any():
            limit = np.atleast_1d(self.K/self.r / (t + self.K/(self.r*E0)))
            if limit.ndim < E.ndim:
                limit = limit[..., None]
            E[mask] = np.broadcast_to(limit, E.shape)[mask]
        return E.reshape(t.shape)

    def deriv(
        self,
        E: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Implicit time derivative for solving ODEs.

        Parameters
        ----------
        E0, h
            Initial state and harvesting rate.
            Must be mutually broadcastable.
        """
        E, h = np.broadcast_arrays(E, h)
        return self.r*E*(1-E/self.K) - h*E

    def tpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable.
        """
        Et = self(t, E0, h)
        return self.deriv(Et, h)

    def hpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        h: float | FloatND = 0
    ) -> float | FloatND:
        """Partial derivative with respect to harvesting rate.

        Parameters
        ----------
        t, E0, h
            Time, initial state of the environment and harvesting rate(s).
            Must be jointly broadcastable.
        """
        t, E0, h = np.broadcast_arrays(t, E0, h)
        rh, Kh = self.adjust_params(self.r, self.K, h)
        D = (Kh-E0)/E0
        D = D.reshape(rh.shape)
        with catch_warnings():
            filterwarnings("ignore", "overflow|invalid|divide")
            X   = np.exp(-rh*t)
            L   = D*X
            L2  = (L+1)**2
            dE1 = -self.K/(self.r*(L+1))
            dE2 = -Kh*t*L / L2
            dE3 =  Kh*(self.K/(E0*self.r)*X) / L2
            dE = np.atleast_1d(dE1 + dE2 + dE3)
        mask = np.isclose(rh, 0, **self.limtol)
        if mask.any():
            mask = np.broadcast_to(mask, dE.shape)
            r = self.r
            K = self.K
            num = -E0*K*t*(2*K+E0*r*t)
            denom = 2*(K + E0*r*t)**2
            limit = np.atleast_1d(num / denom)
            if limit.ndim < dE.ndim:
                limit = limit[..., None]
            dE[mask] = np.broadcast_to(limit, dE.shape)[mask]
        return dE.reshape(t.shape)

    # Internals ------------------------------------------------------------------------

    @staticmethod
    def adjust_params(
        r: float,
        K: float,
        h: float | FloatND
    ) -> tuple[float | FloatND, float | FloatND]:
        """Get harvesting-adjusted environment model parameters."""
        r, K, h = make_arrays(r, K, h)
        return r-h, (1-h/r)*K

    @staticmethod
    def get_r_from_timescale(T: float, epsilon: float = _default_epsilon) -> float:
        r"""Get growth rate :math:`r` from the characteristic timescale
        :math:`T_{\epsilon}`.
        """
        return -2/T*np.log(epsilon/(1-epsilon))

    def get_params(self) -> tuple[FloatND, FloatND]:
        return make_arrays(self.r, self.K)
