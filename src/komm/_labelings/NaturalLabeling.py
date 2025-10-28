from functools import cache
from typing import cast

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.bit_operations import bits_to_int, int_to_bits
from .base import BitBasedLabeling


class NaturalLabeling(BitBasedLabeling, abc.Labeling):
    r"""
    Natural binary labeling. It is a [binary labeling](/ref/Labeling) in which integer $i \in [0 : 2^m)$ is mapped to its base-$2$ representation (MSB-first).
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._num_bits})"

    @property
    @cache
    def matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.NaturalLabeling(2)
            >>> labeling.matrix
            array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
        """
        m = self._num_bits
        ints = np.arange(2**m, dtype=int)
        return int_to_bits(ints, width=m, bit_order="MSB-first")

    def indices_to_bits(self, indices: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.NaturalLabeling(2)
            >>> labeling.indices_to_bits([2, 0])
            array([1, 0, 0, 0])
            >>> labeling.indices_to_bits([[2, 0], [3, 3]])
            array([[1, 0, 0, 0],
                   [1, 1, 1, 1]])
        """
        m = self._num_bits
        indices = np.asarray(indices, dtype=int)
        bits = int_to_bits(indices, width=m, bit_order="MSB-first")
        return bits.reshape(*indices.shape[:-1], -1)

    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.NaturalLabeling(2)
            >>> labeling.bits_to_indices([1, 0, 0, 0])
            array([2, 0])
            >>> labeling.bits_to_indices([[1, 0, 0, 0], [1, 1, 1, 1]])
            array([[2, 0],
                   [3, 3]])
        """
        m = self._num_bits
        bits = np.asarray(bits, dtype=int)
        indices = bits_to_int(bits.reshape(-1, m), bit_order="MSB-first")
        indices = cast(npt.NDArray[np.integer], indices)
        return indices.reshape(*bits.shape[:-1], -1)

    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> labeling = komm.NaturalLabeling(2)
            >>> labeling.marginalize([0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25])
            array([-0.84729786, -0.40546511,  0.        ,  0.        ])
            >>> labeling.marginalize([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
            array([[-0.84729786, -0.40546511],
                   [ 0.        ,  0.        ]])
        """
        return super().marginalize(metrics)
