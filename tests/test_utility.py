import pytest
import numpy as np
from gxr.envir.functions.utility import UtilLinSqrt


class TestUtilLinSqrt:

    @pytest.fixture(scope="module", params=[.1, 1, 5])
    def util(self, request) -> UtilLinSqrt:
        b = request.param
        return UtilLinSqrt(b)

    @pytest.mark.parametrize("seed", [1, 10, 1000])
    def test_integration(self, util: UtilLinSqrt, seed: int) -> None:
        np.random.seed(seed)
        mu = 0
        sigma = 5
        shape = (20,)
        X0 = np.random.normal(mu, sigma, shape)
        X1 = np.random.normal(mu, sigma, shape)
        X  = np.linspace(X0, X1, 20000)
        U0 = util(X0)
        U1 = util(X1)
        dU = util.deriv(X)
        U  = U0 + np.trapz(dU, x=X, axis=0)
        assert np.allclose(U, U1)
