# ruff: noqa: PT003
import numpy as np
import pytest

from gxr.envir.functions import Envir
from gxr.envir.functions.accumulation import Accumulation
from gxr.typing import FloatND
from gxr.utils.array import ndgrid

from .conftest import FunctionTester


@pytest.fixture(scope="function")
def accumulation_Eh(envir_Eh) -> tuple[Accumulation, FloatND, FloatND]:
    envir, E0, h = envir_Eh
    return Accumulation(envir), E0, h


@pytest.fixture(scope="function")
def accumulation_tEh(envir_tEh) -> tuple[Accumulation, FloatND, FloatND, FloatND]:
    envir, t, E0, h = envir_tEh
    return Accumulation(envir), t, E0, h


class TestAccumulation(FunctionTester):
    def test_call_broadcasting(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        A = accumulation(t, E0, h)
        assert A.shape == np.broadcast(t, E0, h).shape

    def test_continuity_correction(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, _ = accumulation_tEh
        h = accumulation.envir.r
        At = accumulation(t, E0, h)
        Al1 = accumulation(t, E0, h - 1e-6)
        Al2 = accumulation(t, E0, h + 1e-6)
        assert np.allclose(At, Al1)
        assert np.allclose(At, Al2)
        assert np.allclose(Al1, Al2, **self.tol)

    def test_tpartial_broadcasting(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        dA = accumulation.tpartial(t, E0, h)
        assert dA.shape == np.broadcast(t, E0, h).shape

    def test_tpartial_integration(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        E0, h = np.broadcast_arrays(E0, h)
        T = np.linspace(0, t, 500).squeeze()
        T = ndgrid(T, E0)[0]
        dA = accumulation.tpartial(T, E0, h)
        At = accumulation(T[-1], E0, h)
        A0 = accumulation(T[0], E0, h)
        A = A0 + np.trapz(dA.T, x=T.T, axis=-1).T
        assert np.allclose(A, At, **self.tol)

    def test_hpartial_broadcasting(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        dA = accumulation.hpartial(t, E0, h)
        assert dA.shape == np.broadcast(t, E0, h).shape

    def test_hpartial_integration(
        self, accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        H = np.linspace(0, h, 100)
        H = np.swapaxes(H, 0, -1)
        T = ndgrid(t, E0)[0]
        dA = accumulation.hpartial(T[..., None], E0[..., None], H)
        Ah = accumulation(T, E0, H[..., -1])
        Ah0 = accumulation(T, E0, H[..., 0])
        A = Ah0 + np.trapz(dA, x=H, axis=-1)
        assert np.allclose(A, Ah, **self.tol)

    def test_gradient_integration(
        self, accumulation_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        t = t.max()
        h = h.mean()
        T, dT = np.linspace(0, t, 500, retstep=True)
        H, dH = np.linspace(0, h, 500, retstep=True)
        A0 = accumulation(np.zeros_like(t), E0, np.zeros_like(h))
        At = accumulation(t, E0, h)
        D = np.stack([dT, dH], axis=0)
        gA = accumulation.gradient(T, E0[..., None], H)
        gA = (gA[..., :-1] + gA[..., 1:]) / 2
        A = A0 + (gA.T * D).T.sum(axis=(0, -1))
        assert np.allclose(A, At, **self.tol)

    def test_tderiv_integration(
        self, accumulation_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        h = np.atleast_1d(h)
        T = np.full_like(h, t.max())
        T = np.linspace(0, T, 100)
        H = np.linspace(0, h, 100)
        dA = accumulation.tderiv(T, E0, H)
        A0 = accumulation(T[0], E0, H[0])
        At = accumulation(T[-1], E0, H[-1])
        A = A0 + np.trapz(dA, x=T, axis=0)
        assert np.allclose(A, At, **self.tol)
