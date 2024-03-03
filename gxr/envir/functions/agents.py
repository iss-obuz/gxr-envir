# ruff: noqa: RET504
from typing import Any

import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND
from gxr.utils.array import expand_dims

from .abc import AgentsFunction
from .accumulation import Accumulation
from .envir import Envir
from .utility import UtilIdentity, UtilityFunction


@registry.envir.functions.register("profits")
class Profits(AgentsFunction):
    """ "Profits function.

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
        self, envir: Envir, sustenance: float = 0.0, cost: float = 0.0, **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.envir = envir
        self.sustenance = sustenance
        self.cost = cost

    @property
    def accumulation(self) -> Accumulation:
        return Accumulation(self.envir)

    def __call__(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Profits.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        t, E0, H, h = self.prepare_data(H, t, E0)
        A = self.accumulation(t, E0, h)
        V = H * A
        C = t * (H * self.cost + self.sustenance)
        P = V - C
        return P

    def deriv(self, E: FloatND, H: FloatND) -> FloatND:
        """Implicit time derivative for solving ODEs.

        Parameters
        ----------
        E, H
            Environment states and individual harvesting rates.
            Must be broadcastable in the arguments' order.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        E, H = self.prepare_data(H, E, h=False)
        dP = H * (E - self.cost) - self.sustenance
        return dP

    def tpartial(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        t, E0, H, h = self.prepare_data(H, t, E0)
        dA = self.accumulation.tpartial(t, E0, h)
        dP = H * dA - H * self.cost - self.sustenance
        return dP

    def hpartial(self, t: FloatND, E0: FloatND, H: FloatND) -> tuple[FloatND, FloatND]:
        """Partial derivatives with respect to agents' own harvesting rates
        and harvesting rates of another agent.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        t, E0, H, h = self.prepare_data(H, t, E0)
        A = self.accumulation(t, E0, h)
        dA = self.accumulation.hpartial(t, E0, h)
        dPj = H * dA
        dPi = dPj + A - self.cost * t
        return dPi, dPj

    def gradient(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Gradient.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        return self._gradient(t, E0, H=H)

    def tderiv(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H[0]`` must be broadcastable in the arguments' order.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
            Moreover, ``t`` and ``H[..., 0]`` must be of the same shape.
        """
        return self._tderiv(t, H, E0, _time_dependent=True)

    # Internals ------------------------------------------------------------------------

    def rescale_cost_rates(self, n_agents: int, envir: Envir | None = None) -> None:
        """Scale sustenance and harvesting cost rates."""
        if envir is None:
            envir = self.envir
        self.sustenance *= envir.r * envir.K / (4 * n_agents)
        self.cost *= envir.K / (2 * n_agents)


@registry.envir.functions.register("foresight")
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

    _default_epsilon = 0.01

    def __init__(
        self,
        profits: Profits,
        horizon: float = 1.0,
        epsilon: float = _default_epsilon,
        **kwds: Any,
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
        return self.epsilon ** (1 / (self.horizon * self.envir.T_epsilon))

    @property
    def tmax(self) -> float:
        """Effective endpoint of the foresight time interval."""
        return np.log(self.epsilon) / np.log(self.gamma)

    def __call__(self, E0: FloatND, H: FloatND) -> FloatND:
        """Foresight function.

        Parameters
        ----------
        E0, H
            Initial state of the environment and individual harvesting rates.
            ``E0`` and ``H`` must be broadcastable in the order of arguments.
            ``H.sum(axis=-1)`` must give overall harvesting rate(s).
        """
        E0, H, h = self.prepare_data(H, E0)
        T = self.make_T(H)
        W = self.make_W(T)
        Et = self.envir(T, E0, h)
        dP = np.atleast_1d(self.profits.deriv(Et, H))
        if H.size == 1:
            dP = dP[..., 0]
        F = np.trapz(W * dP, x=T, axis=0)
        return F

    def tpartial(self, E0: FloatND, H: FloatND) -> FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
            ``H.sum(axis=0)`` must give overall rates.
        """
        E0, _ = self.prepare_data(H, E0, h=False)
        return np.zeros(E0.shape)

    def hpartial(self, E0: FloatND, H: FloatND) -> tuple[FloatND, FloatND]:
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
        E0, H, h = self.prepare_data(H, E0)
        T = self.make_T(H)
        W = self.make_W(T)
        Et = self.envir(T, E0, h)
        dE = self.envir.hpartial(T, E0, h)
        dFj = H * dE
        dFi = Et - self.profits.cost + dFj
        dFi = np.trapz(W * dFi, x=T, axis=0)
        dFj = np.trapz(W * dFj, x=T, axis=0)
        return dFi, dFj

    def gradient(self, E0: FloatND, H: FloatND) -> FloatND:
        """Gradient.

        Parameters
        ----------
        E0
            Initial state. Must be broadcastable with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
            ``H.sum(axis=0)`` must give overall rates.
        """
        return self._gradient(E0, H=H, _time_dependent=False)

    def tderiv(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
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

    def make_T(self, H: FloatND) -> FloatND:
        """Make time grid for foresight."""
        H = np.array(H)
        start = np.zeros(H.shape[-1])
        T = self.make_grid(start, self.tmax)
        T = expand_dims(T, H.ndim - 1, axis=1)
        return T

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
        Core utility function. Defaults to identity function,
    """

    def __init__(
        self, foresight: Foresight, func: UtilityFunction | None = None, **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.foresight = foresight
        self.func = func or UtilIdentity()

    @property
    def envir(self) -> Envir:
        return self.foresight.envir

    def __call__(self, E0: FloatND, H: FloatND) -> FloatND:
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

    def tpartial(self, E0: FloatND, H: FloatND) -> FloatND:
        """Partial derivative with respect to time.

        Parameters
        ----------
        E0
            Initial state. Must be broadcasting with ``H.sum(axis=0)``.
        H
            Individual harvesting rates.
            ``H.sum(axis=0)`` must give overall rates.
        """
        E0, _ = self.prepare_data(H, E0, h=False)
        return np.zeros(E0.shape)

    def hpartial(self, E0: FloatND, H: FloatND) -> tuple[FloatND, FloatND]:
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
        H = np.atleast_1d(H)
        E0, _ = np.broadcast_arrays(E0, H[0])
        dFi, dFj = self.foresight.hpartial(E0, H)
        if not self.func.is_identity:
            F = self.foresight(E0, H)
            dU = self.func.deriv(F)
            dFi *= dU
            dFj *= dU
        return dFi, dFj

    def gradient(self, E0: FloatND, H: FloatND) -> FloatND:
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
        return self._gradient(E0, H=H, _time_dependent=False)

    def tderiv(self, t: FloatND, E0: FloatND, H: FloatND) -> FloatND:
        """Time path derivative.

        Parameters
        ----------
        t, E0, H
            Time, initial state of the environment and individual harvesting rates.
            ``t``, ``E0`` and ``H[0]`` must be broadcastable in the arguments' order.
            ``H.sum(axis=0)`` must give overall harvesting rate(s).
        """
        return self._tderiv(t, H, E0, _time_dependent=False)
