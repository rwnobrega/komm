from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ..types import Array2D

T = TypeVar("T", np.floating, np.complexfloating)


class Labeling(ABC):
    @property
    @abstractmethod
    def matrix(self) -> Array2D[np.integer]:
        r"""
        The labeling matrix $\mathbf{Q}$.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_bits(self) -> int:
        r"""
        The number $m$ of bits per index of the labeling.
        """
        return self.matrix.shape[1]

    @property
    @abstractmethod
    def cardinality(self) -> int:
        r"""
        The cardinality $2^m$ of the labeling.
        """
        return self.matrix.shape[0]

    @property
    @abstractmethod
    def inverse_mapping(self) -> dict[tuple[int, ...], int]:
        r"""
        The inverse mapping of the labeling. It is a dictionary that maps each binary tuple to the corresponding index.
        """
        return {tuple(map(int, row)): i for i, row in enumerate(self.matrix)}

    @abstractmethod
    def indices_to_bits(self, indices: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Returns the binary representation of the given indices.

        Parameters:
            indices: The indices to be converted to bits. Must be an array of integers in $[0:2^m)$.

        Returns:
            bits: The binary representations of the given indices. Has the same shape as `indices`, but with the last dimension expanded by a factor of $m$.
        """
        M = self.cardinality
        indices = np.asarray(indices, dtype=int)
        if not (np.all(indices >= 0) and np.all(indices < M)):
            raise ValueError(f"elements of 'index' must be in [0:{M})")
        bits = self.matrix[indices].reshape(*indices.shape[:-1], -1)
        return bits

    @abstractmethod
    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Returns the indices corresponding to a given sequence of bits.

        Parameters:
            bits: The bits to be converted to indices. Must be an array with elements in $\mathbb{B}$ whose last dimension is a multiple $m$.

        Returns:
            indices: The indices corresponding to the given bits. Has the same shape as `bits`, but with the last dimension contracted by a factor of $m$.
        """
        m = self.num_bits
        bits = np.asarray(bits, dtype=int)
        if bits.shape[-1] % m != 0:
            raise ValueError(
                "last dimension of 'bits' must be a multiple of the number of"
                f" bits per index {m} (got {bits.shape[-1]})"
            )
        if not np.all(np.isin(bits, [0, 1])):
            raise ValueError("elements of 'bits' must be either 0 or 1")
        indices = np.apply_along_axis(
            func1d=lambda row: self.inverse_mapping[tuple(row)],
            axis=-1,
            arr=bits.reshape(*bits.shape[:-1], -1, m),
        ).reshape(*bits.shape[:-1], -1)
        return indices

    @abstractmethod
    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Marginalize metrics over the bits of the labeling. The metrics may represent likelihoods or probabilities, for example. The marginalization is done by computing the L-values of the bits, which are defined as
        $$
        L(\mathtt{b}_i) = \log \frac{\Pr[\mathtt{b}_i = 0]}{\Pr[\mathtt{b}_i = 1]}.
        $$

        Parameters:
            metrics: The metrics for each index of the labeling. Must be an array whose last dimension is a multiple of $2^m$.

        Returns:
            lvalues: The marginalized metrics over the bits of the labeling. Has the same shape as `metrics`, but with the last dimension changed by a factor of $m / 2^m$.
        """
        m, M = self.num_bits, self.cardinality
        metrics = np.asarray(metrics)
        if metrics.shape[-1] % M != 0:
            raise ValueError(
                "last dimension of 'metrics' must be a multiple of the cardinality"
                f" {M} of the labeling (got {metrics.shape[-1]})"
            )
        mask0 = self.matrix == 0
        mask1 = self.matrix == 1
        m = metrics.reshape(*metrics.shape[:-1], -1, M)
        with np.errstate(divide="ignore"):
            lvalues = np.log(m @ mask0) - np.log(m @ mask1)
        return lvalues.reshape(*metrics.shape[:-1], -1)
