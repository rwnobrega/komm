from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .progress import tqdm_bar

T = TypeVar("T", bound=np.generic)
U = TypeVar("U", bound=np.generic)


ArrayFunction = Callable[[npt.NDArray[T]], npt.NDArray[U]]


def vectorize(desc: str):
    r"""
    Vectorizes a function that accepts a 1D-array and returns a 1D-array. Shows a progress bar with description `desc`.
    """

    def decorator(func: ArrayFunction[T, U]) -> ArrayFunction[T, U]:
        @chunkwise(1, desc)
        @wraps(func)
        def wrapper(rows: npt.NDArray[T]) -> npt.NDArray[U]:
            return func(rows[0])[np.newaxis]

        return wrapper

    return decorator


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


def chunkwise(chunk_size: int, desc: str):
    r"""
    Vectorizes a function that accepts a 2D-array and returns a 2D-array, acting row by row. Calls it on chunks of `chunk_size` rows. Shows a progress bar with description `desc`.
    """

    def decorator(func: ArrayFunction[T, U]) -> ArrayFunction[T, U]:
        @wraps(func)
        def wrapper(arr: npt.NDArray[T]) -> npt.NDArray[U]:
            rows = arr.reshape(-1, arr.shape[-1])
            pbar = tqdm_bar(rows.shape[0], desc, unit="block")
            outputs: list[npt.NDArray[U]] = []
            for start in range(0, rows.shape[0], chunk_size):
                chunk = rows[start : start + chunk_size]
                outputs.append(func(chunk))
                pbar.update(chunk.shape[0])
            pbar.close()
            output = np.concatenate(outputs)
            return output.reshape(*arr.shape[:-1], output.shape[-1])

        return wrapper

    return decorator
