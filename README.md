# GXREnvir: environmental games and ABMs for GuestXR project

## Overview

```bash

```

## Installation

```bash
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git
# From a specific branch, e.g. 'dev'
pip install git+ssh://git@github.com/iss-obuz/gxr-envir.git@dev
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
