from itertools import product
import pytest
import numpy as np
from gxr.typing import Float1D, FloatND
from gxr.utils.array import ndgrid, make_arrays
from gxr.envir.functions import Envir, Accumulation
from gxr.envir.functions import Profits, Foresight, Utility
from gxr.envir.functions.utility import UtilIdentity, UtilLinSqrt


@pytest.fixture(scope="function", params=[
    (20, 20), (20, 25), (20, 30), (50, 30)
])
def envir(request) -> Envir:
    return Envir(*request.param)

@pytest.fixture(scope="function", params=[
    1, [0, 1, 2], [[0, 1, 2], [2, 5, 8]]
])
def t(request) -> FloatND:
    return np.array(request.param)

@pytest.fixture(scope="function", params=[
    [.6, .4],
    [.6, [.2, .9]], [[.6, .3], .5],
    [[.6, 1.2], [1, .8]],
    [[[.5, .4], [.3, .2]], [[.5, .7], [2, .1]]],
    [[1, .1], [[.1], [.5]]]
])
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

@pytest.fixture(scope="function")
def accumulation_Eh(envir_Eh) -> tuple[Accumulation, FloatND, FloatND]:
    envir, E0, h = envir_Eh
    return Accumulation(envir), E0, h
@pytest.fixture(scope="function")
def accumulation_tEh(envir_tEh) -> tuple[Accumulation, FloatND, FloatND, FloatND]:
    envir, t, E0, h = envir_tEh
    return Accumulation(envir), t, E0, h

@pytest.fixture(scope="function", params=list(product(
    [1, 4], [(0, 0), (0, .2), (.2, 0), (.2, .1)]
)))
def profits_EH(accumulation_Eh, request) -> tuple[Profits, FloatND, FloatND]:
    accumulation, E0, h = accumulation_Eh
    n_agents, sc = request.param
    sustenance, cost = sc
    I = np.ones(n_agents)
    H = (h[None, ...].T * (I/I.size)).T
    profits = Profits(accumulation, sustenance, cost)
    profits.rescale_cost_rates(n_agents, accumulation.envir)
    return profits, E0, H
@pytest.fixture(scope="function")
def profits_tEH(profits_EH, t) -> tuple[Profits, FloatND, FloatND, FloatND]:
    profits, E0, H = profits_EH
    t = ndgrid(t, np.broadcast_arrays(E0, H[0])[0])[0]
    return profits, t, E0, H

@pytest.fixture(scope="function", params=[.8, .5])
def foresight_EH(profits_EH, request) -> tuple[Foresight, FloatND, FloatND]:
    profits, E0, H = profits_EH
    gamma = request.param
    foresight = Foresight(profits, gamma=gamma)
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
    def get_tspan(T: float | Float1D) -> tuple[float, float]:
        if isinstance(T, np.ndarray) and T.size > 1:
            return T.min(), T.max()
        return 0, T*1
