import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

PathLike = str | bytes | os.PathLike
Figure = Figure
Axes = Axes
AxesGrid = np.ndarray[tuple[int, ...], Axes]

Int1D = np.ndarray[tuple[int], np.integer]
Int2D = np.ndarray[tuple[int, int], np.integer]
IntND = np.ndarray[tuple[int, ...], np.integer]
Float1D = np.ndarray[tuple[int], np.floating]
Float2D = np.ndarray[tuple[int, int], np.floating]
FloatND = np.ndarray[tuple[int, ...], np.floating]

Array1D = np.ndarray[tuple[int]]
Array2D = np.ndarray[tuple[int, int]]
ArrayND = np.ndarray[tuple[int, ...]]
Numeric = float | FloatND | int | IntND
