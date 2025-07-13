from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class ASKConstellation(abc.Constellation[np.complexfloating]):
    r"""
    Amplitude-shift keying (ASK) constellation. It is a complex one-dimensional [constellation](/ref/Constellation) in which the symbols are *uniformly arranged* in a ray. More precisely, the $i$-th symbol is given by
    $$
        x_i = iA \exp(\mathrm{j} 2 \pi \phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order*, $A$ is the *base amplitude*, and $\phi$ is the *phase offset* of the constellation.

    Parameters:
        order: The order $M$ of the constellation.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation (in turns, not radians). The default value is `0.0`.

    Examples:
        1. The $4$-ASK constellation with base amplitude $A = 1$ and phase offset $\phi = 0$ is depicted below
            <figure markdown>
            ![4-ASK constellation.](/fig/constellation_ask_4.svg)
            </figure>

                >>> const = komm.ASKConstellation(4)

        1. The $4$-ASK constellation with base amplitude $A = 2\sqrt{2}$ and phase offset $\phi = 1/8$ is depicted below.
            <figure markdown>
            ![4-ASK constellation.](/fig/constellation_ask_4_turn.svg)
            </figure>

                >>> const = komm.ASKConstellation(
                ...     order=4,
                ...     base_amplitude=2 * np.sqrt(2),
                ...     phase_offset=1 / 8,
                ... )
    """

    def __init__(
        self, order: int, base_amplitude: float = 1.0, phase_offset: float = 0.0
    ) -> None:
        self._order = order
        self._base_amplitude = np.float64(base_amplitude)
        self._phase_offset = np.float64(phase_offset)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"base_amplitude={self._base_amplitude}",
            f"phase_offset={self._phase_offset}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.matrix
            array([[0.+0.j],
                   [1.+0.j],
                   [2.+0.j],
                   [3.+0.j]])
        """
        M, A, φ = self._order, self._base_amplitude, self._phase_offset
        i = np.arange(M).reshape(-1, 1)
        matrix = A * i * np.exp(2j * np.pi * φ)
        return matrix

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the ASK constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.complexfloating]:
        r"""
        For uniform priors, the mean of the ASK constellation is given by
        $$
            \mathbf{m} = \frac{A}{2} (M-1) \exp(\mathrm{j}\phi).
        $$

        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.mean()
            array([1.5+0.j])
        """
        if priors is None:
            M, A, φ = self._order, self._base_amplitude, self._phase_offset
            m = 0.5 * A * (M - 1) * np.exp(2j * np.pi * φ)
            return np.array([m], dtype=complex)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the ASK constellation is given by
        $$
            E = \frac{A^2}{6} (M - 1) (2M - 1).
        $$

        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.mean_energy()
            np.float64(3.5)
        """
        if priors is None:
            M, A = self._order, self._base_amplitude
            return A**2 * (M - 1) * (2 * M - 1) / 6
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the ASK constellation, the minimum distance is given by
        $$
            d_{\min} = A
        $$

        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.minimum_distance()
            np.float64(1.0)
        """
        return self._base_amplitude

    def indices_to_symbols(
        self, indices: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.indices_to_symbols([3, 0])
            array([3.+0.j, 0.+0.j])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.closest_indices([3.1 + 0.2j, 0.1 - 0.2j])
            array([3, 0])
        """
        M, A, φ = self._order, self._base_amplitude, self._phase_offset
        normalized = np.real(np.multiply(received, np.exp(-2j * np.pi * φ))) / A
        indices = np.round(normalized).astype(int)
        indices = np.clip(indices, 0, M - 1)
        return indices

    def closest_symbols(
        self, received: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.closest_symbols([3.1 + 0.2j, 0.1 - 0.2j])
            array([3.+0.j, 0.+0.j])
        """
        return super().closest_symbols(received)

    def posteriors(
        self,
        received: npt.ArrayLike,
        snr: float,
        priors: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.ASKConstellation(4)
            >>> const.posteriors([3.1 + 0.2j, 0.1 - 0.2j], snr=5.0).round(3)
            array([0.   , 0.002, 0.152, 0.846, 0.755, 0.241, 0.004, 0.   ])
        """
        return super().posteriors(received, snr, priors)
