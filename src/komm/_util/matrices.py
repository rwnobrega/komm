from typing import Any

import numpy as np
import numpy.typing as npt

DType = np.number[Any]


def _cartesian_product_2d(
    A: npt.NDArray[DType], B: npt.NDArray[DType]
) -> npt.NDArray[DType]:
    rA, cA = A.shape
    rB, cB = B.shape
    C = np.empty((rA * rB, cA + cB), dtype=np.result_type(A, B))
    for i, rowA in enumerate(A):
        for j, rowB in enumerate(B):
            C[j * rA + i, :] = np.concatenate((rowA, rowB))
    return C


def cartesian_product(
    A: npt.NDArray[DType], B: npt.NDArray[DType]
) -> npt.NDArray[DType]:
    r"""
    Computes the Cartesian product of two vectors/matrices.
    See SA15, eq. (2.2), where it is called the 'ordered direct product' and uses a different convention.
    """
    if A.ndim != B.ndim:
        raise ValueError("A and B must have the same number of dimensions")
    if A.ndim == 1 and B.ndim == 1:
        return _cartesian_product_2d(A.reshape(-1, 1), B.reshape(-1, 1))
    if A.ndim == 2 and B.ndim == 2:
        return _cartesian_product_2d(A, B)
    raise ValueError("A and B must be vectors or matrices")
