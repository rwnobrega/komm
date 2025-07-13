from functools import cache
from itertools import product
from math import prod
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .. import abc
from .Labeling import Labeling

T = TypeVar("T", bound=np.generic)


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
                   [0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 0],
                   [1, 0, 1],
                   [1, 0, 0],
                   [0, 1, 1],
                   [0, 1, 0]])
        """
        matrices = [lab.matrix for lab in self._labelings]
        rows = [np.hstack(comb) for comb in product(*matrices)]
        return np.vstack(rows)

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
             (0, 0, 0): 1,
             (1, 1, 1): 2,
             (1, 1, 0): 3,
             (1, 0, 1): 4,
             (1, 0, 0): 5,
             (0, 1, 1): 6,
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
            array([1, 1, 1, 0, 0, 1])
        """
        return super().indices_to_bits(indices)

    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.bits_to_indices([1, 1, 1, 0, 0, 1])
            array([2, 0])
        """
        return super().bits_to_indices(bits)

    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r""" """
        return super().marginalize(metrics)
