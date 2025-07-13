from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class PSKConstellation(abc.Constellation[np.complexfloating]):
    r"""
    Phase-shift keying (PSK) constellation. It is a complex one-dimensional [constellation](/ref/Constellation) in which the symbols are *uniformly arranged* in a circle. More precisely, the $i$-th symbol is given by
    $$
        x_i = A \exp \left( \mathrm{j} \frac{2 \pi i}{M} \right) \exp(\mathrm{j} 2 \pi \phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order*, $A$ is the *amplitude*, and $\phi$ is the *phase offset* of the constellation.

    Parameters:
        order: The order $M$ of the constellation.

        amplitude: The amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation (in turns, not radians). The default value is `0.0`.

    Examples:
        1. The $4$-PSK constellation with amplitude $A = 1$ and phase offset $\phi = 0$ is depicted below.
            <figure markdown>
            ![4-PSK constellation.](/fig/constellation_psk_4.svg)
            </figure>

                >>> const = komm.PSKConstellation(4)

        1. The $8$-PSK constellation with amplitude $A = 0.5$ and phase offset $\phi = 1/16$ is depicted below.
            <figure markdown>
            ![8-PSK constellation.](/fig/constellation_psk_8.svg)
            </figure>

                >>> const = komm.PSKConstellation(8, amplitude=0.5, phase_offset=1 / 16)
    """

    def __init__(
        self, order: int, amplitude: float = 1.0, phase_offset: float = 0.0
    ) -> None:
        self._order = order
        self._amplitude = np.float64(amplitude)
        self._phase_offset = np.float64(phase_offset)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"amplitude={self._amplitude}",
            f"phase_offset={self._phase_offset}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.matrix
            array([[ 1.+0.j],
                   [ 0.+1.j],
                   [-1.+0.j],
                   [ 0.-1.j]])
        """
        M, A, φ = self._order, self._amplitude, self._phase_offset
        i = np.arange(M).reshape(-1, 1)
        matrix = A * np.exp(2j * np.pi * i / M) * np.exp(2j * np.pi * φ)
        # We round to avoid sin(pi) != 0, and add 0.0 to avoid -0.0.
        return matrix.round(15) + 0.0

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the PSK constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.complexfloating]:
        r"""
        For uniform priors, the mean of the PSK constellation is given by
        $$
            \mathbf{m} = 0.
        $$

        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.mean()
            array([0.+0.j])
        """
        if priors is None:
            return np.zeros((1,), dtype=complex)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the PSK constellation is given by
        $$
            E = A^2.
        $$

        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.mean_energy()
            np.float64(1.0)
        """
        if priors is None:
            A = self._amplitude
            return A**2
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the PSK constellation, the minimum distance is given by
        $$
            d_\mathrm{min} = 2A\sin\left(\frac{\pi}{M}\right).
        $$

        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.minimum_distance()
            np.float64(1.414213562373095)
        """
        M, A = self._order, self._amplitude
        return 2 * A * np.sin(np.pi / M)

    def indices_to_symbols(
        self, indices: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.indices_to_symbols([3, 0])
            array([0.-1.j, 1.+0.j])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.closest_indices([0.1 - 1.1j, 1.2 + 0.1j])
            array([3, 0])
        """
        M, φ = self._order, self._phase_offset
        turn = np.angle(np.multiply(received, np.exp(-2j * np.pi * φ))) / (2 * np.pi)
        indices = np.mod(np.around(M * turn), M).astype(int)
        return indices

    def closest_symbols(
        self, received: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.PSKConstellation(4)
            >>> const.closest_symbols([0.1 - 1.1j, 1.2 + 0.1j])
            array([0.-1.j, 1.+0.j])
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
            >>> const = komm.PSKConstellation(4)
            >>> const.posteriors([0.1 - 1.1j, 1.2 + 0.1j], snr=2.0).round(3)
            array([0.018, 0.   , 0.008, 0.974, 0.982, 0.012, 0.   , 0.005])
        """
        return super().posteriors(received, snr, priors)
