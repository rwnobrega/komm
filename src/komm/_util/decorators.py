from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)
U = TypeVar("U", bound=np.generic)


def vectorized_method(
    func: Callable[..., npt.NDArray[U]],
) -> Callable[..., npt.NDArray[U]]:
    r"""
    Decorator to vectorize a method that accepts a 1D array and returns a 1D array. The decorator will apply the method along the last axis of a multidimensional array. The decorated method should have the following signature:

    ```python
    def method(self, x: np.ndarray[T], *args, **kwargs) -> np.ndarray[U]:
        ...
    ```
    """

    @wraps(func)
    def wrapper(
        self: object,
        arr: npt.NDArray[T],
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray[U]:
        def func1d(x: npt.NDArray[T]) -> npt.NDArray[U]:
            return func(self, x, *args, **kwargs)

        return np.apply_along_axis(func1d, -1, arr)

    return wrapper
