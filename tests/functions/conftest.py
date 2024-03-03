# ruff: noqa: PT003
from itertools import product

import numpy as np
import pytest

from gxr.envir.functions import Envir, Foresight, Profits, Utility
from gxr.envir.functions.utility import UtilIdentity, UtilLinSqrt
from gxr.typing import FloatND
from gxr.utils.array import make_arrays, ndgrid


@pytest.fixture(scope="function", params=[(20, 20), (20, 25), (20, 30), (50, 30)])
def envir(request) -> Envir:
    return Envir(*request.param)


@pytest.fixture(scope="function", params=[1, [0, 1, 2], [[0, 1, 2], [2, 5, 8]]])
def t(request) -> FloatND:
    return np.array(request.param)


@pytest.fixture(
    scope="function",
    params=[
        [0.6, 0.4],
        [0.6, [0.2, 0.9]],
        [[0.6, 0.3], 0.5],
        [[0.6, 1.2], [1, 0.8]],
        [[[0.5, 0.4], [0.3, 0.2]], [[0.5, 0.7], [2, 0.1]]],
        [[1, 0.1], [[0.1], [0.5]]],
    ],
)
def Eh(request) -> tuple[FloatND, FloatND]:
    return make_arrays(*request.param)


@pytest.fixture(scope="function")
def envir_Eh(envir, Eh) -> tuple[Envir, FloatND, FloatND]:
    E, h = Eh
    E *= envir.K
    h *= envir.r
    return envir, E, h


@pytest.fixture(scope="function")
def envir_tEh(envir_Eh, t) -> tuple[Envir, FloatND, FloatND, FloatND]:
    envir, E, h = envir_Eh
    t = ndgrid(t, np.broadcast_arrays(E, h)[0])[0]
    return envir, t, E, h


@pytest.fixture(
    scope="function",
    params=list(product([1, 4], [(0, 0), (0, 0.2), (0.2, 0), (0.2, 0.1)])),
)
def profits_EH(envir_Eh, request) -> tuple[Profits, FloatND, FloatND]:
    envir, E0, h = envir_Eh
    n_agents, sc = request.param
    sustenance, cost = sc
    I = np.ones(n_agents)  # noqa
    H = h[..., None] * (I / I.size)
    profits = Profits(envir, sustenance, cost)
    profits.rescale_cost_rates(n_agents)
    E0, _ = np.broadcast_arrays(E0, H[..., 0])
    E0 = E0[..., None]
    return profits, E0, H


@pytest.fixture(scope="function")
def profits_tEH(profits_EH, t) -> tuple[Profits, FloatND, FloatND, FloatND]:
    profits, E0, H = profits_EH
    t = ndgrid(t, np.broadcast_arrays(E0, H[0])[0])[0]
    return profits, t, E0, H


@pytest.fixture(scope="function", params=[1, 0.5])
def foresight_EH(profits_EH, request) -> tuple[Foresight, FloatND, FloatND]:
    profits, E0, H = profits_EH
    horizon = request.param
    foresight = Foresight(profits, horizon=horizon)
    return foresight, E0, H


@pytest.fixture(scope="function", params=[UtilIdentity(), UtilLinSqrt(1)])
def utility_EH(foresight_EH, request) -> tuple[Utility, FloatND, FloatND]:
    func = request.param
    foresight, E, H = foresight_EH
    utility = Utility(foresight, func=func)
    return utility, E, H


class FunctionTester:
    @property
    def tol(self) -> dict[str, float]:
        return {"rtol": 1e-2, "atol": 1e-2}

    @staticmethod
    def get_tspan(T: float | FloatND) -> tuple[float, float]:
        if isinstance(T, np.ndarray) and T.size > 1:
            return T.min(), T.max()
        return 0, T * 1

    @staticmethod
    def make_agent_shape(H: FloatND, *args: FloatND) -> tuple[int, ...]:
        shape = np.broadcast(*args, H).shape
        if H.size == 1 and any(x.size > 1 for x in args):
            shape = (*shape, 1)
        return shape
