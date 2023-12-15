"""Various functions used in environmental game."""
from typing import Self, Mapping
from abc import abstractmethod
import numpy as np
from .abc import DifferentiableFunction
from ...typing import FloatND
from ...utils.misc import obj_from_dict


class UtilityFunction(DifferentiableFunction):
    """Differentiable 1D utility function based on agent's profit.

    Function must be defined through the ``self.__call__`` method,
    and its derivative in ``self.deriv``. All function parameters
    has to be passed during initialization.
    """
    @property
    def is_identity(self) -> bool:
        return False

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        """Construct from a mapping."""
        return obj_from_dict(dct, package=cls.__module__)


class UtilLinear(UtilityFunction):
    """Linear utility function.

    Attributes
    ----------
    a
        Constant coefficient.
    b
        Slope coefficient.
    """
    def __init__(self, a: float = 0, b: float = 1) -> None:
        self.a = a
        self.b = b

    def __call__(self, x: float | FloatND) -> FloatND:
        return self.a + self.b*x

    def deriv(self, x: float | FloatND) -> FloatND:
        return np.full_like(x, self.b)

    def is_identity(self) -> bool:
        return self.a == 0 and self.b == 1


class UtilIdentity(UtilLinear):
    """Identity utility function."""
    def __init__(self) -> None:
        super().__init__(a=0, b=1)


class UtilLinSqrt(UtilityFunction):
    """Linear-square root utility function.

    Attributes
    ----------
    b
        Derivative around zero.
        Has to be positive.
    """
    def __init__(self, b: float = 1) -> None:
        if b <= 0:
            raise ValueError("'b' cannot be negative")
        self.b = b

    def __call__(self, x: float | FloatND) -> float | FloatND:
        if np.isscalar(x):
            x = np.array([x])
        u = np.empty_like(x, dtype=float)
        m = x >= 0
        u[m] = 2*np.sqrt(x[m] + 1/self.b**2) - 2/self.b
        m = ~m
        u[m] = self.b*x[m]
        return u

    def deriv(self, x: float | FloatND) -> float | FloatND:
        """Derivative function."""
        if np.isscalar(x):
            x = np.array([x])
        u = np.empty_like(x, dtype=float)
        m = x >= 0
        u[m] = 1 / np.sqrt(x[m] + 1/self.b**2)
        u[~m] = self.b
        return u
