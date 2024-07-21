"""Various functions used in environmental game."""

import numpy as np

from gxr.envir.config import registry
from gxr.typing import FloatND

from .abc import DifferentiableFunction


class UtilityFunction(DifferentiableFunction):
    """Differentiable 1D utility function based on agent's profit.

    Function must be defined through the ``self.__call__`` method,
    and its derivative in ``self.deriv``. All function parameters
    has to be passed during initialization.
    """

    @property
    def is_identity(self) -> bool:
        return False


@registry.envir.utility.register("linear")
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
        return self.a + self.b * x

    def deriv(self, x: float | FloatND) -> FloatND:
        return np.full_like(x, self.b)

    def is_identity(self) -> bool:
        return self.a == 0 and self.b == 1


@registry.envir.utility.register("identity")
class UtilIdentity(UtilLinear):
    """Identity utility function."""

    def __init__(self) -> None:
        super().__init__(a=0, b=1)


@registry.envir.utility.register("linsqrt")
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
            errmsg = "'b' cannot be negative"
            raise ValueError(errmsg)
        self.b = b

    def __call__(self, x: float | FloatND) -> float | FloatND:
        x = np.atleast_1d(x)
        u = np.empty_like(x, dtype=float)
        m = x >= 0
        u[m] = 2 * np.sqrt(x[m] + 1 / self.b**2) - 2 / self.b
        m = ~m
        u[m] = self.b * x[m]
        return u

    def deriv(self, x: float | FloatND) -> float | FloatND:
        """Derivative function."""
        x = np.atleast_1d(x)
        u = np.empty_like(x, dtype=float)
        m = x >= 0
        u[m] = 1 / np.sqrt(x[m] + 1 / self.b**2)
        u[~m] = self.b
        return u


@registry.envir.utility.register("linroot")
class UtilLinRoot(UtilityFunction):
    """Linear-arbitrary root utility function.

    Attributes
    ----------
    b
        Derivative around zero.
        Has to be positive.
    c
        Root order for positive values.
        Has to be positive.
    """

    def __init__(self, c: float = 2, b: float = 1) -> None:
        self.b = b
        self.c = c

    def __call__(self, x: float | FloatND) -> float | FloatND:
        b = self.b
        c = self.c
        x = np.atleast_1d(x)
        y = b * x
        m = x > 0
        if m.any():
            mu = 1 if c == 1 else c ** (1 / (1 - c))
            y[m] = b * ((x[m] + mu**c) ** (1 / c) - mu)
        return y

    def deriv(self, x: float | FloatND) -> float | FloatND:
        """Derivative function."""
        b = self.b
        c = self.c
        x = np.atleast_1d(x)
        y = np.full_like(x, b)
        m = x <= 0
        if m.any():
            y[m] = b / c * (x[m] + c ** (c / (1 - c))) ** ((1 - c) / c)
        return y
