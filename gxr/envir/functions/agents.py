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
    """
    def __init__(
        self,
        envir: Envir,
        sustenance: float = 0.0,
        cost: float = 0.0,
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        h = self.make_h(H)
        t, E0, H, h = np.broadcast_arrays(t, E0, H, h)
        A = self.accumulation(t, E0, h)
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
            Must be broadcastable in the arguments' order.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        E, H = np.broadcast_arrays(E, H)
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        h = self.make_h(H)
        t, E0, H, h = np.broadcast_arrays(t, E0, H, h)
        dA = self.accumulation.tpartial(t, E0, h)
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        h = self.make_h(H)
        t, E0, H, h = np.broadcast_arrays(t, E0, H, h)
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H[0]`` must be broadcastable in the arguments' order.
            ``H.sum(axis=0)`` must give overall harvesting rate(s).
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
    horizon
        Width of the foresight expressed in terms of the number
        of characteristic timescales of the environment.
    epsilon
        :math:`\epsilon`-threshold for determining
        the characteristic timescale of foresight.
    """
    def __init__(
        self,
        profits: Profits,
        horizon: float = 1.0,
        epsilon: float = .01,
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.profits = profits
        self.horizon = horizon
        self.epsilon = epsilon

    @property
    def envir(self) -> Envir:
        return self.profits.envir

    @property
    def gamma(self) -> float:
        """Foresight discount rate."""
        return self.epsilon**(1/(self.horizon*self.envir.T_epsilon))

    @property
    def tmax(self) -> float:
        """Effective endpoint of the foresight time interval."""
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
            ``H.sum(axis=0)`` must give overall rates.
        """
        H, E0 = self.align_with_H(H, E0)
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
            ``H.sum(axis=0)`` must give overall rates.
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
            ``H.sum(axis=0)`` must give overall rates.
        """
        H = np.array(H)
        E0, _ = np.broadcast_arrays(E0, H[0])
        T  = self.make_T(E0.shape)
        H, T, E0 = self.align_with_H(H, T, E0)
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
            ``H.sum(axis=0)`` must give overall rates.
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H[0]`` must be broadcastable in the arguments' order.
            ``H.sum(axis=0)`` must give overall harvesting rate(s).
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
            ``H.sum(axis=0)`` must give overall rates.
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
            ``H.sum(axis=0)`` must give overall rates.
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
            ``H.sum(axis=0)`` must give overall rates.
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
            ``H.sum(axis=0)`` must give overall rates.
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
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H[0]`` must be broadcastable in the arguments' order.
            ``H.sum(axis=0)`` must give overall harvesting rate(s).
        """
        return self._tderiv(t, H, E0, _time_dependent=False)
