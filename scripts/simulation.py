from pathlib import Path

import numpy as np  # noqa

from gxr.envir.simulation import Simulation

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PARAMS = {
    "params.K": [100],
    "params.n_agents": [4],
    "params.sustenance": [0.4],
    "params.horizon": np.linspace(0.01, 5, 11),
    "params.alpha": np.linspace(0, 1, 11),
    "params.c": [2],
    "params.delay": [0.5],
    "params.bias": np.linspace(0, 1, 11),
}

simulation = Simulation(
    n_epochs=10,
    n_reps=20,
    params=PARAMS,
    seed=3034283429,
    n_jobs=20,
    split_epochs=True,
)

simulation.run(DATA / "simulation.parquet")
