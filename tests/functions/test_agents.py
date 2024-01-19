import numpy as np
from scipy.integrate import solve_ivp
from gxr.typing import FloatND
from gxr.utils.array import ndgrid
from gxr.envir.functions import Profits, Foresight, Utility
from .conftest import FunctionTester


class TestProfits(FunctionTester):

    def test_call_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        P = profits(t, E0, H)
        shape = np.broadcast(t, E0, H[..., 0]).shape
        assert P.shape == shape or (1,)

    def test_deriv_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, _, E0, H = profits_tEH
        dP = profits.deriv(E0, H)
        shape = np.broadcast(E0, H[..., 0]).shape
        assert dP.shape == shape or (1,)

    def test_ode(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        def ode(t, y, H):
            return np.array([
                profits.envir.deriv(y[0], H.sum(axis=-1)),
                *profits.deriv(y[0], H)
            ])
        profits, t, E0, H = profits_tEH
        t_span = (0, t.max()+.1)
        E0 = E0.mean()
        if H.ndim > 1:
            H = H.mean(axis=tuple(range(1, H.ndim)))
        h  = H.sum(axis=0)
        Et = profits.envir(t_span[1], E0, h)
        P0 = profits(t_span[0], E0, H)
        Pt = profits(t_span[1], E0, H)
        y0 = np.array([E0, *P0])
        sol = solve_ivp(ode, t_span, y0, args=(H,))
        assert np.allclose(sol.y[0, 0], E0, **self.tol)
        assert np.allclose(sol.y[0, -1], Et, **self.tol)
        assert np.allclose(sol.y[1:, 0], P0, **self.tol)
        assert np.allclose(sol.y[1:, -1], Pt, **self.tol)

    def test_tpartial_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        dP = profits.tpartial(t, E0, H)
        shape = np.broadcast(t, E0, H[0]).shape
        assert dP.shape == shape or (1,)

    def test_tpartial_integration(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        T  = np.linspace(0, t, 500).squeeze()
        T  = ndgrid(T, E0)[0]
        dP = profits.tpartial(T, E0, H)
        P0 = profits(T[0], E0, H)
        Pt = profits(T[-1], E0, H)
        P  = P0 + np.trapz(dP, x=T, axis=0)
        assert np.allclose(P, Pt, **self.tol)

    def test_hpartial_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        dPi, dPj = profits.hpartial(t, E0, H)
        shape = np.broadcast(t, E0, H[0]).shape
        assert dPi.shape == dPj.shape == shape or (1,)

    def test_gradient_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        gP = profits.gradient(t, E0, H)
        n_agents = H.shape[-1]
        shape = np.broadcast(t, E0).shape
        shape = (*shape[:-1], n_agents+1, n_agents)
        assert gP.shape == shape

    def test_gradient_integration(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        E0 = E0.mean()
        T, dT = np.linspace(0, t.max(), 500, retstep=True)
        H, dH = np.linspace(0, H.mean(axis=tuple(range(H.ndim-1))), 500, retstep=True)
        D  = np.array([dT, *dH])
        P0 = profits(T[0], E0, H[0])
        Pt = profits(T[-1], E0, H[-1])
        gP = profits.gradient(T[..., None], E0, H)
        dP = D[..., None]*gP
        dP = (dP[:-1] + dP[1:]) / 2
        P  = P0 + dP.sum(axis=(0, 1))
        assert np.allclose(P, Pt, **self.tol)

    def test_tderiv_broadcasting(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        E0 = E0.mean()
        t  = np.linspace(0, t.max(), 10)
        H  = np.linspace(0, H.mean(axis=tuple(range(H.ndim-1))), 10)
        dP = profits.tderiv(t, E0, H)
        shape = (*np.broadcast(t, E0).shape, H.shape[-1])
        assert dP.shape == shape

    def test_tderiv_integration(
        self,
        profits_tEH: tuple[Profits, FloatND, FloatND, FloatND]
    ) -> None:
        profits, t, E0, H = profits_tEH
        E0 = E0.mean()
        t  = np.linspace(0, t.max(), 500)
        H  = np.linspace(0, H.mean(axis=tuple(range(H.ndim-1))), 500)
        P0 = profits(t[0], E0, H[0])
        Pt = profits(t[-1], E0, H[-1])
        dP = profits.tderiv(t, E0, H)
        P  = P0 + np.trapz(dP, x=t[..., None], axis=0)
        assert np.allclose(P, Pt, **self.tol)


class TestForesight(FunctionTester):

    def test_call_broadcasting(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        F = foresight(E0, H)
        shape = (len(H), *np.broadcast(E0, H[0]).shape)
        assert F.shape == shape

    def test_tpartial_broadcasting(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        F = foresight(E0, H)
        dF = foresight.tpartial(E0, H)
        assert np.all(dF == 0) and F.shape == dF.shape

    def test_hpartial_broadcasting(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        F = foresight(E0, H)
        dFi, dFj = foresight.hpartial(E0, H)
        assert F.shape == dFi.shape == dFj.shape

    def test_gradient_broadcasting(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        gF = foresight.gradient(E0, H)
        n_agents = len(H)
        shape = (n_agents, 1+n_agents, *np.broadcast(E0, H[0]).shape)
        assert gF.shape == shape

    def test_gradient_integration(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        E0 = E0.mean()
        ng = 20
        H, dH = np.linspace(0, H.mean(axis=tuple(range(1, H.ndim))), ng, retstep=True)
        H  = np.moveaxis(H, 0, -1)
        D  = np.array([*dH])
        F0 = foresight(E0, H[..., 0])
        Ft = foresight(E0, H[..., -1])
        gF = foresight.gradient(E0, H)
        dF = (D*gF.T).T
        dF = (dF[..., :-1] + dF[..., 1:]) / 2
        F  = F0 + dF.sum(axis=(1, -1))
        assert np.allclose(F, Ft, **self.tol)

    def test_tderiv_broadcasting(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        E0 = E0.mean()
        ng = 5
        t  = np.linspace(0, 10, ng)
        H  = np.linspace(0, H.mean(axis=tuple(range(1, H.ndim))), ng)
        H  = np.moveaxis(H, 0, -1)
        dF = foresight.tderiv(t, E0, H)
        shape = (len(H), *np.broadcast(t, E0, H[0]).shape)
        assert dF.shape == shape

    def test_tderiv_integration(
        self,
        foresight_EH: tuple[Foresight, FloatND, FloatND]
    ) -> None:
        foresight, E0, H = foresight_EH
        E0 = E0.mean()
        if H.ndim > 1:
            H = H.mean(axis=tuple(range(1, H.ndim)))
        ng = 20
        t  = np.linspace(0, 10, ng)
        H  = np.linspace(0, H, ng)
        H  = np.moveaxis(H, 0, -1)
        F0 = foresight(E0, H[..., 0])
        Ft = foresight(E0, H[..., -1])
        dF = foresight.tderiv(t, E0, H)
        F  = F0 + np.trapz(dF, x=t, axis=1)
        assert np.allclose(F, Ft, **self.tol)


class TestUtility(FunctionTester):

    def test_call_broadcasting(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        U = utility(E0, H)
        shape = (len(H), *np.broadcast(E0, H[0]).shape)
        assert U.shape == shape

    def test_tpartial_broadcasting(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        U = utility(E0, H)
        dU = utility.tpartial(E0, H)
        assert np.all(dU == 0) and U.shape == dU.shape

    def test_hpartial_broadcasting(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        U = utility(E0, H)
        dUi, dUj = utility.hpartial(E0, H)
        assert U.shape == dUi.shape == dUj.shape

    def test_gradient_broadcasting(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        gU = utility.gradient(E0, H)
        n_agents = len(H)
        shape = (n_agents, 1+n_agents, *np.broadcast(E0, H[0]).shape)
        assert gU.shape == shape

    def test_gradient_integration(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        E0 = E0.mean()
        ng = 25
        H, dH = np.linspace(0, H.mean(axis=tuple(range(1, H.ndim))), ng, retstep=True)
        H  = np.moveaxis(H, 0, -1)
        D  = np.array([*dH])
        U0 = utility(E0, H[..., 0])
        Ut = utility(E0, H[..., -1])
        gU = utility.gradient(E0, H)
        dU = (D*gU.T).T
        dU = (dU[..., :-1] + dU[..., 1:]) / 2
        U  = U0 + dU.sum(axis=(1, -1))
        assert np.allclose(U, Ut, **self.tol)

    def test_tderiv_broadcasting(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        E0 = E0.mean()
        ng = 5
        t  = np.linspace(0, 10, ng)
        H  = np.linspace(0, H.mean(axis=tuple(range(1, H.ndim))), ng)
        H  = np.moveaxis(H, 0, -1)
        dU = utility.tderiv(t, E0, H)
        shape = (len(H), *np.broadcast(t, E0, H[0]).shape)
        assert dU.shape == shape

    def test_tderiv_integration(
        self,
        utility_EH: tuple[Utility, FloatND, FloatND]
    ) -> None:
        utility, E0, H = utility_EH
        E0 = E0.mean()
        if H.ndim > 1:
            H = H.mean(axis=(tuple(range(1, H.ndim))))
        ng = 25
        t  = np.linspace(0, 10, ng)
        H  = np.linspace(0, H.mean(axis=tuple(range(1, H.ndim))), ng)
        H  = np.moveaxis(H, 0, -1)
        U0 = utility(E0, H[..., 0])
        Ut = utility(E0, H[..., -1])
        dU = utility.tderiv(t, E0, H)
        U  = U0 + np.trapz(dU, x=t, axis=1)
        assert np.allclose(U, Ut, **self.tol)
