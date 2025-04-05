from typing import Any

import numpy as np
import numpy.typing as npt


def array(data: Any, etype: type) -> npt.NDArray[Any]:
    array = np.asarray(data)
    result = np.empty(array.shape, dtype=object)
    for idx in np.ndindex(array.shape):
        result[idx] = etype(array[idx])
    return result
