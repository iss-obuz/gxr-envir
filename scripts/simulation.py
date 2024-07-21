from pathlib import Path

from gxr.envir.simulation import Simulation

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PARAMS = {
    "params.K": [100],
    "params.n_agents": [1, 4, 10],
    "params.sustenance": [0, 0.25, 0.5, 0.75, 0.9],
    "params.horizon": [0.01, 0.5, 1, 2, 5, 10],
    "params.alpha": [0, 0.2, 0.5, 0.7, 1],
    "params.c": [1 / 2, 1, 2, 4],
    "params.delay": [0.1, 1, 3],
    "params.noise": [0.001, 0.05, 0.01, 0.03, 0.05, 0.1, 0.5],
}

simulation = Simulation(
    n_epochs=20,
    n_reps=10,
    params=PARAMS,
    seed=3034283429,
    n_jobs=20,
)

simulation.run(DATA / "simulation.parquet")
