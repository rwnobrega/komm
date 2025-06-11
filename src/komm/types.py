from typing import TypeVar

import numpy as np

DType = TypeVar("DType", bound=np.generic)

Array1D = np.ndarray[tuple[int], np.dtype[DType]]
Array2D = np.ndarray[tuple[int, int], np.dtype[DType]]
