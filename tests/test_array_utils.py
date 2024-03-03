from collections.abc import Iterable

import numpy as np
import pytest

from gxr.typing import FloatND
from gxr.utils.array import expand_dims, ndgrid, numderiv


@pytest.mark.parametrize(
    ("axis", "n", "expected"),
    [
        (1, 1, (5, 1, 5, 10, 4)),
        (0, 2, (1, 1, 5, 5, 10, 4)),
        (-1, 3, (5, 5, 10, 4, 1, 1, 1)),
    ],
)
def test_expand_dims(axis: int, n: int, expected: tuple[int, ...]) -> None:
    X = np.ones((5, 5, 10, 4))
    assert expand_dims(X, n, axis).shape == expected


@pytest.mark.parametrize(
    ("Xs", "expected"),
    [
        ((), ()),
        ([(10,), (5,)], [(10, 1), (1, 5)]),
        ([(), (2,)], [(), (2,)]),
        ([(), (2, 3), (7)], [(), (2, 3, 1), (1, 1, 7)]),
    ],
)
def test_ndgrid(Xs: Iterable[np.ndarray], expected: Iterable[tuple[int, ...]]) -> None:
    Xs = ndgrid(*(np.ones(shape) for shape in Xs))
    result = [X.shape for X in Xs]
    assert result == list(expected)


@pytest.mark.parametrize(
    ("X", "Y", "axis"),
    [
        (np.linspace(10, 100, 100), np.linspace(0, 1, 100), -1),
    ],
)
def test_numderiv(X: FloatND, Y: FloatND, axis: int) -> None:
    grad = numderiv(X, Y, axis=axis)
    exp = np.gradient(X, Y, axis=axis)
    assert np.allclose(grad, exp)
