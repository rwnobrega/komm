from functools import cache
from math import prod
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .. import abc
from .Labeling import Labeling

T = TypeVar("T", bound=np.generic)


def cartesian_product(A: npt.NDArray[T], B: npt.NDArray[T]) -> npt.NDArray[T]:
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
    for i, rowA in enumerate(A):
        for j, rowB in enumerate(B):
            product[j * rA + i, :] = np.concatenate((rowA, rowB))
    return product


class ProductLabeling(abc.Labeling):
    r"""
    Cartesian product of labelings.

    Parameters:
        *labelings: Labelings to be combined. At least one labeling is required.
    """

    def __init__(self, *labelings: abc.Labeling) -> None:
        if len(labelings) < 1:
            raise ValueError("at least one labeling is required")
        self._labelings = labelings

    @classmethod
    def from_matrices(cls, *matrices: npt.ArrayLike) -> "ProductLabeling":
        return cls(*(Labeling(m) for m in matrices))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._labelings})"

    @property
    @cache
    def matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.matrix
            array([[0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 1, 0],
                   [1, 0, 0],
                   [0, 1, 0]])
        """
        matrix = self._labelings[0].matrix
        for labeling in self._labelings[1:]:
            matrix = cartesian_product(matrix, labeling.matrix)
        return matrix

    @property
    def num_bits(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.num_bits
            3
        """
        return sum(lab.num_bits for lab in self._labelings)

    @property
    def cardinality(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.cardinality
            8
        """
        return prod(lab.cardinality for lab in self._labelings)

    @property
    @cache
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.inverse_labeling
            {(0, 0, 1): 0,
             (1, 1, 1): 1,
             (1, 0, 1): 2,
             (0, 1, 1): 3,
             (0, 0, 0): 4,
             (1, 1, 0): 5,
             (1, 0, 0): 6,
             (0, 1, 0): 7}
        """
        return super().inverse_labeling

    def indices_to_bits(self, indices: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.indices_to_bits([2, 0])
            array([1, 0, 1, 0, 0, 1])
        """
        return super().indices_to_bits(indices)

    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.bits_to_indices([1, 0, 1, 0, 0, 1])
            array([2, 0])
        """
        return super().bits_to_indices(bits)

    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r""" """
        return super().marginalize(metrics)
