from functools import cache
from math import isqrt

import numpy as np
import numpy.typing as npt

from .. import abc
from .ProductLabeling import ProductLabeling
from .ReflectedLabeling import ReflectedLabeling


class ReflectedRectangularLabeling(abc.Labeling):
    r"""
    Reflected rectangular binary labeling. It is the [Cartesian product](/ref/ProductLabeling) of two [reflected binary labelings](/ref/ReflectedLabeling), possibly with distinct number of bits.
    """

    def __init__(
        self,
        num_bits: int | tuple[int, int],
        _pre_cache: bool = True,
    ) -> None:
        if isinstance(num_bits, int):
            if not (num_bits > 0 and isqrt(num_bits) ** 2 == num_bits):
                raise ValueError(
                    "if a single integer, 'num_bits' must be a positive perfect square"
                )
            num_bits = (isqrt(num_bits), isqrt(num_bits))
        if not (num_bits[0] >= 1 and num_bits[1] >= 1):
            raise ValueError("'num_bits' must contain positive integers")
        self._num_bits = num_bits
        if _pre_cache:
            self.matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._num_bits})"

    @property
    @cache
    def matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4)
            >>> labeling.matrix
            array([[0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 1, 1, 1],
                   [0, 1, 1, 0],
                   [1, 1, 0, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 1],
                   [1, 0, 1, 0]])
        """
        mi, mq = self._num_bits
        matrix = ProductLabeling(ReflectedLabeling(mi), ReflectedLabeling(mq)).matrix
        return matrix

    @property
    def num_bits(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4)
            >>> labeling.num_bits
            4
        """
        mi, mq = self._num_bits
        return mi + mq

    @property
    def cardinality(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4)
            >>> labeling.cardinality
            16
        """
        mi, mq = self._num_bits
        return 2 ** (mi + mq)

    @property
    # @cache
    def inverse_mapping(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4  )
            >>> labeling.inverse_mapping
            {(0, 0, 0, 0): 0,
             (0, 0, 0, 1): 1,
             (0, 0, 1, 1): 2,
             (0, 0, 1, 0): 3,
             (0, 1, 0, 0): 4,
             (0, 1, 0, 1): 5,
             (0, 1, 1, 1): 6,
             (0, 1, 1, 0): 7,
             (1, 1, 0, 0): 8,
             (1, 1, 0, 1): 9,
             (1, 1, 1, 1): 10,
             (1, 1, 1, 0): 11,
             (1, 0, 0, 0): 12,
             (1, 0, 0, 1): 13,
             (1, 0, 1, 1): 14,
             (1, 0, 1, 0): 15}
        """
        return super().inverse_mapping

    def indices_to_bits(self, indices: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4)
            >>> labeling.indices_to_bits([8, 13])
            array([1, 1, 0, 0, 1, 0, 0, 1])
            >>> labeling.indices_to_bits([[8, 13], [0, 1]])
            array([[1, 1, 0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 1]])
        """
        return super().indices_to_bits(indices)

    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.ReflectedRectangularLabeling(4)
            >>> labeling.bits_to_indices([1, 1, 0, 0, 1, 0, 0, 1])
            array([ 8, 13])
            >>> labeling.bits_to_indices([
            ...     [1, 1, 0, 0, 1, 0, 0, 1],
            ...     [0, 0, 0, 0, 0, 0, 0, 1],
            ... ])
            array([[ 8, 13],
                   [ 0,  1]])
        """
        return super().bits_to_indices(bits)

    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return super().marginalize(metrics)
