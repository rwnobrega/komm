from typing import Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt

from .._util.bit_operations import int_to_bits

T = TypeVar("T", bound=npt.NDArray[np.number])


def cartesian_product(A: T, B: T) -> T:
    r"""
    Computes the Cartesian product of two matrices. See <cite>SA15, eq. (2.2)</cite>, where it is called the 'ordered direct product' and uses a different convention.

    Parameters:
        A: First input matrix, with shape (rA, cA).
        B: Second input matrix, with shape (rB, cB).

    Returns:
        The Cartesian product matrix, with shape (rA * rB, cA + cB)
    """
    rA, cA = A.shape
    rB, cB = B.shape
    product = np.zeros((rA * rB, cA + cB), dtype=A.dtype)
    product = cast(T, product)
    for i, rowA in enumerate(A):
        for j, rowB in enumerate(B):
            product[j * rA + i, :] = np.concatenate((rowA, rowB))
    return product


def labeling_natural(order: int) -> npt.NDArray[np.integer]:
    m = order.bit_length() - 1
    labeling = np.empty((order, m), dtype=int)
    for i in range(order):
        labeling[i, :] = int_to_bits(i, m)[::-1]
    return labeling


def labeling_reflected(order: int) -> npt.NDArray[np.integer]:
    m = order.bit_length() - 1
    labeling = np.empty((order, m), dtype=int)
    for i in range(order):
        labeling[i, :] = int_to_bits(i ^ (i >> 1), m)[::-1]
    return labeling


def labeling_natural_2d(orders: tuple[int, int]) -> npt.NDArray[np.integer]:
    order_I, order_Q = orders
    return cartesian_product(
        labeling_natural(order_I),
        labeling_natural(order_Q),
    )


def labeling_reflected_2d(orders: tuple[int, int]) -> npt.NDArray[np.integer]:
    order_I, order_Q = orders
    return cartesian_product(
        labeling_reflected(order_I),
        labeling_reflected(order_Q),
    )


labelings = {
    "natural": labeling_natural,
    "reflected": labeling_reflected,
    "natural_2d": labeling_natural_2d,
    "reflected_2d": labeling_reflected_2d,
}

LabelingStr = Literal["natural", "reflected", "natural_2d", "reflected_2d"]


def get_labeling(
    labeling: LabelingStr | npt.ArrayLike,
    allowed_labelings: tuple[LabelingStr, ...],
    *args: Any,
    **kwargs: Any,
) -> npt.NDArray[np.integer]:
    if isinstance(labeling, str):
        if labeling not in allowed_labelings:
            raise ValueError(f"if string, 'labeling' must be in {allowed_labelings}")
        return labelings[labeling](*args, **kwargs)
    return np.asarray(labeling, dtype=int)
