# GXREnvir: environmental games and ABMs for GuestXR project

## Installation

```bash
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git
# Install from specific version tag, e.g. v0.1
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git@v0.1
```

## Basic usage

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

### Running simulations as systems of differential equations

* A detailed code example of an ODE-based simulation can be found in
  `notebooks/ode-simulations.ipynb`.

### Development

```bash
# Clone
git clone git@github.com:iss-obuz/gxr-envir.git
# Enter the root directory and create Conda environment
cd gxr-envir
conda env create -f environment.yaml
# Install 'gxr.envir' package in editable mode and with dev dependencies
pip install --editable .[dev]
```

### Testing

Unit tests cover only the core workhorse functions, i.e. mostly the
`gxr.envir.functions` module. Jupyter notebooks in `notebooks` directory
provide more end-to-end test as well as documentation and usage examples.

```bash
pytest
## With automatic debugger session
pytest --pdb
```
