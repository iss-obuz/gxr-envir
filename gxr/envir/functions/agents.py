from typing import Any
import numpy as np
from .abc import AgentsFunction
from .envir import Envir
from .accumulation import Accumulation
from .utility import UtilityFunction, UtilIdentity
from ...typing import FloatND


class Profits(AgentsFunction):
    """"Profits function.

    Attributes
    ----------
    envir
        Environment function.
    sustenance
        Agent sustenance rate.
    cost
        Agent harvesting cost rate.
    rKN
        ``r``, ``K`` and ``N`` parameters for natural scaling
        of sustenance and harvesting costs.
    """
    def __init__(
        self,
        envir: Envir,
        sustenance: float = 0,
        cost: float = 0,
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.envir = envir
        self.sustenance = sustenance
        self.cost = cost

    @property
    def accumulation(self) -> Accumulation:
        return Accumulation(self.envir)

    def __call__(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: FloatND
    ) -> float | FloatND:
        """Profits.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0, H
            Initial state and individual harvesting rates.
            ``E0`` has to be broadcastable with ``H.sum(axis=0)``,
            which is expected to return overall harvesting rates
            (sums over agents).
        """
        t, E0, H = self.align_with_H(H, t, E0)
        A = self.accumulation(t, E0, H.sum(axis=0))
        V = H*A
        C = t*(H*self.cost + self.sustenance)
        P = V - C
        return P

    def deriv(self, E: float | FloatND, H: FloatND) -> FloatND:
        """Implicit time derivative for solving ODEs.

        Parameters
        ----------
        E, H
            Environment states and individual harvesting rates.
            Must be mutually broadcastable.
        """
        E, H = self.align_with_H(H, E)
        dP = H*(E - self.cost) - self.sustenance
        return dP

    def tpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND
    ) -> FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0, H
            Initial state and harvesting rate.
            ``E0`` must be broadcastable with ``H.sum(axis=0)``.
        """
        t, E0, H = self.align_with_H(H, t, E0)
        dA = self.accumulation.tpartial(t, E0, H.sum(axis=0))
        dP = H*dA - H*self.cost - self.sustenance
        return dP

    def hpartial(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND,
    ) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect to agents' own harvesting rates
        and harvesting rates of another agent.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0, H
            Initial state and harvesting rate.
            ``E0`` must be broadcastable with ``H.sum(axis=0)``.
        """
        t, E0, H = self.align_with_H(H, t, E0)
        h   = H.sum(axis=0)
        A   = self.accumulation(t, E0, h)
        dA  = self.accumulation.hpartial(t, E0, h)
        dPj = H*dA
        dPi = dPj + A - self.cost*t
        return dPi, dPj

    def gradient(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND,
    ) -> float | FloatND:
        """Gradient.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0, H
            Initial state and harvesting rate.
            ``E0`` must be broadcastable with ``H.sum(axis=0)``.
        """
        return self._gradient(t, E0, H=H)

    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND,
    ) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0, H
            Initial state and harvesting rate.
            ``E0`` must be broadcastable with ``H.sum(axis=0)``.
        """
        return self._tderiv(t, H, E0, _time_dependent=True)

    # Internals ------------------------------------------------------------------------

    def rescale_cost_rates(self, n_agents: int, envir: Envir | None = None) -> None:
        """Scale sustenance and harvesting cost rates."""
        if envir is None:
            envir = self.envir
        self.sustenance *= envir.r*envir.K / (4*n_agents)
        self.cost *= envir.K / (2*n_agents)


class Foresight(AgentsFunction):
    r"""Foresight function.

    Attributes
    ----------
    profits
        Profits function.
    gamma
        Time discount rate.
    epsilon
        :math:`\epsilon`-threshold for determining
        the characteristic timescale of foresight.
    """
    def __init__(
        self,
        profits: Profits,
        gamma: float = .8,
        epsilon: float = .01,
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.profits = profits
        self.gamma = gamma
        self.epsilon = epsilon

    @property
    def envir(self) -> Envir:
        return self.profits.envir

    @property
    def tmax(self) -> float:
        return np.log(self.epsilon) / np.log(self.gamma)

    def __call__(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> float | FloatND:
        """Foresight function.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        H = np.array(H)
        E0, _ = np.broadcast_arrays(E0, H[0])
        T  = self.make_T(E0.shape)
        W  = self.make_W(T)
        Et = self.envir(T, E0, H.sum(axis=0))
        dP = self.profits.deriv(Et, H)
        F  = np.trapz(W*dP, x=T[None, ...], axis=1)
        return F

    def tpartial(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> float | FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        shape = (len(H), *np.broadcast(E0, H[0]).shape)
        return np.zeros(shape)

    def hpartial(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect to agents' own harvesting rates
        and harvesting rates of another agent.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        H = np.array(H)
        E0, _ = np.broadcast_arrays(E0, H[0])
        T  = self.make_T(E0.shape)
        T, E0, H = self.align_with_H(H, T, E0)
        W  = self.make_W(T)
        h  = H.sum(axis=0)
        Et  = self.envir(T, E0, h)
        dE  = self.envir.hpartial(T, E0, h)
        dFj = H*dE
        dFi = Et - self.profits.cost + dFj
        dFi = np.trapz(W*dFi, x=T[None, ...], axis=1)
        dFj = np.trapz(W*dFj, x=T[None, ...], axis=1)
        return dFi, dFj

    def gradient(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> float | FloatND:
        """Gradient.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        return self._gradient(E0, H=H)

    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND
    ) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        return self._tderiv(t, H, E0, _time_dependent=False)

    # Internals ------------------------------------------------------------------------

    def make_T(self, shape: tuple[int, ...] = ()) -> FloatND:
        """Make time grid for foresight."""
        start = np.zeros(shape).squeeze()
        return self.make_grid(start, self.tmax)

    def make_W(self, T: FloatND) -> FloatND:
        """Make discount weights."""
        W = self.gamma**T
        return W / np.trapz(W, x=T, axis=0)


class Utility(AgentsFunction):
    """Agent utility function.

    Attributes
    ----------
    foresight
        Foresight function.
    func
        Core utility function.
    """
    def __init__(
        self,
        foresight: Foresight,
        func: UtilityFunction = UtilIdentity(),
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.foresight = foresight
        self.func = func

    @property
    def envir(self) -> Envir:
        return self.foresight.envir

    def __call__(
        self,
        E0: float | FloatND,
        H: FloatND
    ) -> FloatND:
        """Agent utility function.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        U = self.foresight(E0, H)
        if not self.func.is_identity:
            U = self.func(U)
        return U

    def tpartial(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> float | FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        E0
            Initial state. Must be broadcasting with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        shape = (len(H), *np.broadcast(E0, H[0]).shape)
        return np.zeros(shape)

    def hpartial(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect agents' own harvesting rates
        and harvesting rates of another agent.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        E0, _ = np.broadcast_arrays(E0, H[0])
        dFi, dFj = self.foresight.hpartial(E0, H)
        if not self.func.is_identity:
            F = self.foresight(E0, H)
            dU  = self.func.deriv(F)
            dFi *= dU
            dFj *= dU
        return dFi, dFj

    def gradient(
        self,
        E0: float | FloatND,
        H: float | FloatND
    ) -> float | FloatND:
        """Partial derivatives with respect agents' own harvesting rates
        and harvesting rates of another agent.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        return self._gradient(E0, H=H)

    def tderiv(
        self,
        t: float | FloatND,
        E0: float | FloatND,
        H: float | FloatND
    ) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t
            Time.
            Must be broadcastable with the broadcast of ``E0`` and ``H.sum(axis=0)``.
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
        """
        return self._tderiv(t, H, E0, _time_dependent=False)
