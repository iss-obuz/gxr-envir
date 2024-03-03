"""Array utility functions."""
from typing import Any

import numpy as np

from gxr.typing import FloatND


def make_arrays(*args: Any) -> tuple[np.ndarray, ...]:
    """Convert arguments to arrays."""
    return tuple(np.array(a) for a in args)


def expand_dims(X: np.ndarray, n: int = 1, axis: int = 0) -> np.ndarray:
    """Expand array dimensions.

    Parameters
    ----------
    X
        Array to modify.
    axis
        Axis index at which to start expanding.
    n
        Number of axes to add.
    """
    for _ in range(n):
        X = np.expand_dims(X, axis)
    return X


def ndgrid(*Xs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reshape sequence of arrays or scalars, so they form a grid
    of N-dimensional coordinates allowing for outer vectorization.

    Scalars do not add unitary axes, but are returned
    as 0-dimensional :class:`numpy.ndarray` instances.
    """
    Xs = tuple(np.array(X) for X in Xs)
    ndims = [X.ndim for X in Xs]
    mesh = []
    for i, X in enumerate(Xs):
        if X.ndim >= 1:
            n_left = sum(ndims[:i])
            n_right = sum(ndims[i + 1 :])
            X = expand_dims(X, n_left, 0)
            X = expand_dims(X, n_right, -1)
        mesh.append(X)
    return tuple(mesh)


def numderiv(Y: FloatND, X: FloatND, axis: int = -1) -> FloatND:
    """Numerical gradient approximation.

    Parameters
    ----------
    Y, X
        Arrays of dependent and independent variables.
        Must be of the same shape.
    axis
        Axis along which to differentiate.
    """
    dY = np.diff(Y, axis=axis)
    dX = np.diff(X, axis=axis)
    dYdX = dY / dX
    grad = np.zeros_like(Y)
    grad = np.swapaxes(grad, 0, axis)
    dYdX = np.swapaxes(dYdX, 0, axis)
    grad[:-1] += dYdX
    grad[1:] += dYdX
    grad[1:-1] /= 2
    return np.swapaxes(grad, axis, 0)
