from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_pmf
from ..types import Array1D, Array2D

T = TypeVar("T", np.floating, np.complexfloating)


class Constellation(ABC, Generic[T]):
    @property
    @abstractmethod
    def matrix(self) -> Array2D[T]:
        r"""
        The constellation matrix $\mathbf{X}$.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def order(self) -> int:
        r"""
        The order $M$ of the constellation.
        """
        return self.matrix.shape[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        r"""
        The dimension $N$ of the constellation.
        """
        return self.matrix.shape[1]

    @abstractmethod
    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[T]:
        r"""
        Computes the mean $\mathbf{m}$ of the constellation given prior probabilities $p_i$ of the constellation symbols. It is given by
        $$
        \mathbf{m} = \sum_{i \in [0:M)} p_i \mathbf{x}_i.
        $$

        Parameters:
            priors: The prior probabilities of the constellation symbols. Must be a 1D-array whose size is equal to the order $M$ of the constellation. If not given, uniform priors are assumed.

        Returns:
            mean: The mean $\mathbf{m}$ of the constellation.
        """
        if priors is None:
            priors = np.ones(self.order) / self.order
        priors = validate_pmf(priors)
        return np.dot(priors, self.matrix)

    @abstractmethod
    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        Computes the mean energy $E$ of the constellation given prior probabilities $p_i$ of the constellation symbols. It is given by
        $$
        E = \sum_{i \in [0:M)} p_i \lVert \mathbf{x}_i \rVert^2.
        $$

        Parameters:
            priors: The prior probabilities of the constellation symbols. Must be a 1D-array whose size is equal to the order $M$ of the constellation. If not given, uniform priors are assumed.

        Returns:
            mean_energy: The mean energy $E$ of the constellation.
        """
        if priors is None:
            priors = np.ones(self.order) / self.order
        priors = validate_pmf(priors)
        return priors @ np.sum(np.abs(self.matrix) ** 2, axis=1)

    @abstractmethod
    def minimum_distance(self) -> np.floating:
        r"""
        Computes the minimum Euclidean distance $d_\mathrm{min}$ of the constellation. It is given by
        $$
        d_\mathrm{min} = \min_ { i, j \in [0:M), ~ i \neq j } \lVert \mathrm{x}_i - \mathrm{x}_j \rVert.
        $$
        """
        distances = np.linalg.norm(self.matrix[:, np.newaxis] - self.matrix, axis=2)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        return np.min(distances)

    @abstractmethod
    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Returns the constellation symbols corresponding to the given indices.

        Parameters:
            indices: The indices to be converted to symbols. Must be an array of integers in $[0:M)$.

        Returns:
            symbols: The symbols corresponding to the given indices. Has the same shape as `indices`, but with the last dimension expanded by a factor of $N$.
        """
        indices = np.asarray(indices, dtype=int)
        symbols = self.matrix[indices].reshape(*indices.shape[:-1], -1)
        return symbols

    def _squared_distances(self, input: npt.NDArray[T]) -> npt.NDArray[np.floating]:
        diff = input[..., np.newaxis, :] - self.matrix[np.newaxis, ...]
        return np.sum(np.abs(diff) ** 2, axis=-1)

    @abstractmethod
    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Returns the indices of the constellation symbols closest to the given received points.

        Parameters:
            received: The received points. Must be an array whose last dimension is a multiple of $N$.

        Returns:
            indices: The indices of the symbols closest to the received points. Has the same shape as `received`, but with the last dimension contracted by a factor of $N$.
        """
        N = self.dimension
        received = np.asarray(received)
        if received.shape[-1] % N != 0:
            raise ValueError(
                "last dimension of 'received' must be a multiple of the constellation"
                f" dimension {N} (got {received.shape[-1]})"
            )
        distances = self._squared_distances(received.reshape(-1, N))
        indices = np.argmin(distances, axis=-1).reshape(*received.shape[:-1], -1)
        return indices

    @abstractmethod
    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Returns the constellation symbols closest to the given received points.

        Parameters:
            received: The received points. Must be an array whose last dimension is a multiple of $N$.

        Returns:
            symbols: The symbols closest to the received points. Has the same shape as `received`.
        """
        received = np.asarray(received)
        indices = self.closest_indices(received)
        symbols = self.indices_to_symbols(indices).reshape(*received.shape[:-1], -1)
        return symbols

    @abstractmethod
    def posteriors(
        self,
        received: npt.ArrayLike,
        snr: float,
        priors: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Returns the posterior probabilities of each constellation symbol given received points, the signal-to-noise ratio (SNR), and prior probabilities.

        Parameters:
            received: The received points. Must be an array whose last dimension is a multiple of $N$.

            snr: The signal-to-noise ratio (SNR) of the channel (linear, not decibel).

            priors: The prior probabilities of the symbols. Must be a 1D-array whose size is equal to $M$. If not given, uniform priors are assumed.

        Returns:
            posteriors: The posterior probabilities of each symbol given the received points. Has the same shape as `received`, but with the last dimension changed by a factor of $M / N$.
        """
        M, N = self.order, self.dimension
        if priors is None:
            priors = np.ones(M) / M
        priors = validate_pmf(priors)
        if priors.size != M:
            raise ValueError(
                "length of 'priors' must be equal to the constellation"
                f" order {M} (got {priors.size})"
            )
        received = np.asarray(received)
        if received.shape[-1] % N != 0:
            raise ValueError(
                "last dimension of 'received' must be a multiple of the constellation"
                f" dimension {N} (got {received.shape[-1]})"
            )
        r = received.reshape(-1, N)
        n0 = self.mean_energy(priors) / snr
        distances = self._squared_distances(r)
        logp = -distances / n0 + np.log(priors)
        logp -= np.max(logp, axis=-1, keepdims=True)
        p = np.exp(logp)
        p /= np.sum(p, axis=-1, keepdims=True)
        return p.reshape(*received.shape[:-1], -1)
