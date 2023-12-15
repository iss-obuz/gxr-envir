import numpy as np
from scipy.integrate import solve_ivp
from gxr.typing import FloatND
from gxr.utils.array import ndgrid
from gxr.envir.functions import Envir, Accumulation
from .conftest import FunctionTester


class TestEnvir(FunctionTester):

    def test_call_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        Et = envir(t, E0, h)
        assert Et.shape == np.broadcast(t, E0, h).shape

    def test_continuity_correction(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, _ = envir_tEh
        Et  = envir(t, E0, h=envir.r)
        El1 = envir(t, E0, h=envir.r-1e-6)
        El2 = envir(t, E0, h=envir.r+1e-6)
        assert np.allclose(Et, El1)
        assert np.allclose(Et, El2)
        assert np.allclose(El1, El2, **self.tol)

    def test_deriv_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        dE = envir.deriv(E0, h)
        assert dE.shape == np.broadcast(E0, h).shape
        Et = envir(t, E0, h)
        dE = envir.deriv(Et, h)
        assert Et.shape == dE.shape

    def test_ode(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        def ode(t, y, h):
            return np.atleast_1d(envir.deriv(y, h))

        envir, t, E0, h = envir_tEh
        t_span = (0, t.max()+.1)
        E0 = E0.mean()
        h  = h.mean()
        Et = envir(t_span[1], E0, h)
        y0 = np.array([E0])
        sol = solve_ivp(ode, t_span, y0, args=(h,), max_step=1)
        assert np.allclose(sol.y[0, 0], E0, **self.tol)
        assert np.allclose(sol.y[0, -1], Et, **self.tol)

    def test_tpartial_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        dE = envir.tpartial(t, E0, h)
        assert dE.shape == np.broadcast(t, E0, h).shape

    def test_tpartial_integration(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        E0, h = np.broadcast_arrays(E0, h)
        T   = np.linspace(0, t, 500).squeeze()
        T   = ndgrid(T, E0)[0]
        dE  = envir.tpartial(T, E0, h)
        Et  = envir(T[-1], E0, h)
        Et0 = envir(T[0], E0, h)
        E   = Et0 + np.trapz(dE.T, x=T.T, axis=-1).T
        assert np.allclose(E, Et, **self.tol)

    def test_hpartial_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        dE = envir.hpartial(t, E0, h)
        assert dE.shape == np.broadcast(t, E0, h).shape

    def test_hpartial_continuity_correction(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, _ = envir_tEh
        h = envir.r
        dE = envir.hpartial(t, E0, h)
        dE1 = envir.hpartial(t, E0, h+1e-6)
        dE2 = envir.hpartial(t, E0, h-1e-6)
        assert np.allclose(dE, dE1, **self.tol)
        assert np.allclose(dE, dE2, **self.tol)

    def test_hpartial_integration(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        H   = np.linspace(0, h, 100)
        H   = np.swapaxes(H, 0, -1)
        T = ndgrid(t, E0)[0]
        dE  = envir.hpartial(T[..., None], E0[..., None], H)
        Eh  = envir(T, E0, H[..., -1])
        Eh0 = envir(T, E0, H[..., 0])
        E   = Eh0 + np.trapz(dE, x=H, axis=-1)
        assert np.allclose(E, Eh, **self.tol)

    def test_gradient_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        gE = envir.gradient(t, E0, h)
        shape = (2, *np.broadcast(t, E0, h).shape)
        assert gE.shape == shape

    def test_gradient_integration(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        t = t.max()
        h = h.mean()
        T, dT = np.linspace(0, t, 500, retstep=True)
        H, dH = np.linspace(0, h, 500, retstep=True)
        E0  = envir(np.zeros_like(t), E0, np.zeros_like(h))
        Et  = envir(t, E0, h)
        D   = np.stack([dT, dH], axis=0)
        gE  = envir.gradient(T[:-1], E0[..., None], H[:-1])
        gE += envir.gradient(T[1:], E0[..., None], H[1:])
        gE /= 2
        E   = E0 + (gE.T*D).T.sum(axis=(0, -1))
        assert np.allclose(E, Et, **self.tol)

    def test_tderiv_broadcasting(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        h = np.atleast_1d(h)
        T = np.full_like(h, t.max())
        T = np.linspace(0, T, 100)
        H = np.linspace(0, h, 100)
        E = envir.tderiv(T, E0, H)
        assert E.shape == np.broadcast(E0, H).shape

    def test_tderiv_integration(
        self,
        envir_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        envir, t, E0, h = envir_tEh
        h = np.atleast_1d(h)
        T = np.full_like(h, t.max())
        T = np.linspace(0, T, 100)
        H = np.linspace(0, h, 100)
        dE = envir.tderiv(T, E0, H)
        E0 = envir(T[0], E0, H[0])
        Et = envir(T[-1], E0, H[-1])
        E  = E0 + np.trapz(dE, x=T, axis=0)
        assert np.allclose(E, Et, **self.tol)


class TestAccumulation(FunctionTester):

    def test_call_broadcasting(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        A = accumulation(t, E0, h)
        assert A.shape == np.broadcast(t, E0, h).shape

    def test_continuity_correction(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, _ = accumulation_tEh
        h = accumulation.envir.r
        At  = accumulation(t, E0, h)
        Al1 = accumulation(t, E0, h-1e-6)
        Al2 = accumulation(t, E0, h+1e-6)
        assert np.allclose(At, Al1)
        assert np.allclose(At, Al2)
        assert np.allclose(Al1, Al2, **self.tol)

    def test_tpartial_broadcasting(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        dA = accumulation.tpartial(t, E0, h)
        assert dA.shape == np.broadcast(t, E0, h).shape

    def test_tpartial_integration(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        E0, h = np.broadcast_arrays(E0, h)
        T  = np.linspace(0, t, 500).squeeze()
        T  = ndgrid(T, E0)[0]
        dA = accumulation.tpartial(T, E0, h)
        At = accumulation(T[-1], E0, h)
        A0 = accumulation(T[0], E0, h)
        A  = A0 + np.trapz(dA.T, x=T.T, axis=-1).T
        assert np.allclose(A, At, **self.tol)

    def test_hpartial_broadcasting(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        dA = accumulation.hpartial(t, E0, h)
        assert dA.shape == np.broadcast(t, E0, h).shape

    def test_hpartial_integration(
        self,
        accumulation_tEh: tuple[Accumulation, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        H   = np.linspace(0, h, 100)
        H   = np.swapaxes(H, 0, -1)
        T = ndgrid(t, E0)[0]
        dA  = accumulation.hpartial(T[..., None], E0[..., None], H)
        Ah  = accumulation(T, E0, H[..., -1])
        Ah0 = accumulation(T, E0, H[..., 0])
        A   = Ah0 + np.trapz(dA, x=H, axis=-1)
        assert np.allclose(A, Ah, **self.tol)

    def test_gradient_integration(
        self,
        accumulation_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        t = t.max()
        h = h.mean()
        T, dT = np.linspace(0, t, 500, retstep=True)
        H, dH = np.linspace(0, h, 500, retstep=True)
        A0 = accumulation(np.zeros_like(t), E0, np.zeros_like(h))
        At = accumulation(t, E0, h)
        D  = np.stack([dT, dH], axis=0)
        gA = accumulation.gradient(T, E0[..., None], H)
        gA = (gA[..., :-1] + gA[..., 1:]) / 2
        A  = A0 + (gA.T*D).T.sum(axis=(0, -1))
        assert np.allclose(A, At, **self.tol)

    def test_tderiv_integration(
        self,
        accumulation_tEh: tuple[Envir, FloatND, FloatND, FloatND]
    ) -> None:
        accumulation, t, E0, h = accumulation_tEh
        h = np.atleast_1d(h)
        T = np.full_like(h, t.max())
        T = np.linspace(0, T, 100)
        H = np.linspace(0, h, 100)
        dA = accumulation.tderiv(T, E0, H)
        A0 = accumulation(T[0], E0, H[0])
        At = accumulation(T[-1], E0, H[-1])
        A  = A0 + np.trapz(dA, x=T, axis=0)
        assert np.allclose(A, At, **self.tol)
