from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

T = TypeVar("T", bound=np.generic)
U = TypeVar("U", bound=np.generic)


ArrayFunction = Callable[[npt.NDArray[T]], npt.NDArray[U]]


def vectorize(func: ArrayFunction[T, U]):
    r"""
    Vectorizes a function that accepts a 1D array and returns a 1D array.
    """

    @wraps(func)
    def wrapper(arr: npt.NDArray[T], *args: Any, **kwargs: Any):
        return np.apply_along_axis(func, -1, arr, *args, **kwargs)

    return wrapper


def blockwise(block_size: int):
    r"""
    Applies a function blockwise to the last dimension of an array.
    """

    def decorator(func: ArrayFunction[T, U]):
        @wraps(func)
        def wrapper(arr: npt.ArrayLike):
            arr = np.asarray(arr)
            if arr.shape[-1] % block_size != 0:
                raise ValueError(
                    "last dimension of array must be a multiple of block size"
                    f" {block_size} (got {arr.shape[-1]})"
                )
            blocks = arr.reshape(*arr.shape[:-1], -1, block_size)
            processed = func(blocks)
            return processed.reshape(*processed.shape[:-2], -1)

        return wrapper

    return decorator


def with_pbar(pbar: "tqdm[Any]"):
    r"""
    Updates a given tqdm progress bar after a function call.
    """

    def decorator(func: ArrayFunction[T, U]) -> ArrayFunction[T, U]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            result = func(*args, **kwargs)
            pbar.update()
            return result

        return wrapper

    return decorator
