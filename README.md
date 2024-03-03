# GXREnvir: environmental games and ABMs for GuestXR project

## Installation

```bash
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git
# From a specific branch, e.g. 'dev'
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git@dev
```

## Basic usage

* For running simulations based on ODE system representation see
  `notebooks/1-ode-harvesting.ipynb`.

### Running as a game with externally provided harvesting rates

First, a configuration have to be defined, for instance a `config.toml`
file.

```toml
# config.toml
[profits]
sustenance = .1  # magnitude of the player sustenance rate

[foresight]
horizon = 2   # scaled length of the foresight horizon
alpha = 0     # magnitude of the belief that others will behave like us

[game]
dt = 0.1  # length of a single time step in the game loop

[game.model]
n_agents = 4   # number of agents/players in the game

[game.model.behavior]
delay = 2    # magnitude of the environment perception delay
noise = 0.2  # magnitude of the optimal harvesting change perception noise
```

Then, a game instance can be setup very easily.

```python
from gxr.envir import Config, EnvirGame

config = Config("config.toml")
game = EnvirGame(config["game"])

# Run for 1000 time steps
for _ in range(1000):
    H = ... # This must be supplied by a parent application
    game.step(H)
    # Access state values that can be used in a parent application
    game.E     # state of the environment
    game.Ehat  # perceived state of the environment (including delay)
    game.U     # accumulated player utilities
    game.dH    # perceived optimal change of behavior of players
```

### Dev installation

```bash
pip install "gxr-envir[dev] @ git+ssh://git@github.com/iss-obuz/gxr-envir.git"
```

### Testing

```bash
pytest
## With automatic debugger session
pytest --pdb
```

### Unit test coverage statistics

```bash
# Calculate and display
make coverage
# Only calculate
make cov-run
# Only display (based on previous calculations)
make cov-report
```
