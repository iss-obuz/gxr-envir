from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes as _Axes
from matplotlib.figure import Figure

PathLike = str | Path
Figure = Figure
Axes: TypeAlias = _Axes
AxesGrid = npt.NDArray[np.object_]

IntND = int | npt.NDArray[np.integer[npt.NBitBase]]
FloatND = float | npt.NDArray[np.floating[npt.NBitBase]]
