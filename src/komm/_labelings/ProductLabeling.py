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
    Cartesian product of [labelings](/ref/Labeling).

    Parameters:
        *labelings: The labelings to be combined. At least one labeling is required.

        repeat: Number of times to repeat the full sequence of labelings. Must be a positive integer. Has the same semantics as `itertools.product`. The default value is `1`.

    Examples:
        >>> labeling1 = komm.Labeling([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> labeling2 = komm.Labeling([[1], [0]])
        >>> labeling = komm.ProductLabeling(labeling1, labeling2)
        >>> labeling.matrix
        array([[0, 0, 1],
               [0, 0, 0],
               [1, 1, 1],
               [1, 1, 0],
               [1, 0, 1],
               [1, 0, 0],
               [0, 1, 1],
               [0, 1, 0]])


        >>> labeling = komm.ProductLabeling(
        ...    komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]]),
        ...    repeat=2,
        ... )
        >>> labeling.matrix
        array([[1, 0, 1, 0],
               [1, 0, 1, 1],
               [1, 0, 0, 1],
               [1, 0, 0, 0],
               [1, 1, 1, 0],
               [1, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 1, 0, 0],
               [0, 1, 1, 0],
               [0, 1, 1, 1],
               [0, 1, 0, 1],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 1, 1],
               [0, 0, 0, 1],
               [0, 0, 0, 0]])
    """

    def __init__(self, *labelings: abc.Labeling, repeat: int = 1) -> None:
        if len(labelings) < 1:
            raise ValueError("at least one labeling is required")
        if repeat < 1:
            raise ValueError("'repeat' must be at least 1")
        self._labelings = labelings * repeat

    @classmethod
    def from_matrices(
        cls,
        *matrices: npt.ArrayLike,
        repeat: int = 1,
    ) -> "ProductLabeling":
        r"""
        Constructs a product labeling from labeling matrices.

        Parameters:
            *matrices: The labeling matrices. At least one matrix is required. See [labeling documentation](/ref/Labeling).

            repeat: Number of times to repeat the full sequence of matrices. Must be a positive integer. Has the same semantics as `itertools.product`. The default value is `1`.

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

            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[1, 0], [1, 1], [0, 1], [0, 0]],
            ...     repeat=2,
            ... )
            >>> labeling.matrix
            array([[1, 0, 1, 0],
                   [1, 0, 1, 1],
                   [1, 0, 0, 1],
                   [1, 0, 0, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1],
                   [1, 1, 0, 1],
                   [1, 1, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [0, 1, 0, 1],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
        """
        labelings = tuple(Labeling(m) for m in matrices)
        return cls(*labelings, repeat=repeat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._labelings})"

    @property
    @cache
    def matrix(self) -> npt.NDArray[np.integer]:
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
    def inverse_mapping(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> labeling = komm.ProductLabeling.from_matrices(
            ...     [[0, 0], [1, 1], [1, 0], [0, 1]],
            ...     [[1], [0]],
            ... )
            >>> labeling.inverse_mapping
            {(0, 0, 1): 0,
             (0, 0, 0): 1,
             (1, 1, 1): 2,
             (1, 1, 0): 3,
             (1, 0, 1): 4,
             (1, 0, 0): 5,
             (0, 1, 1): 6,
             (0, 1, 0): 7}
        """
        return super().inverse_mapping

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
