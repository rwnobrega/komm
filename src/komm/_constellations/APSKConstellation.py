from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D
from .PSKConstellation import PSKConstellation


class APSKConstellation(abc.Constellation[np.complexfloating]):
    r"""
    Amplitude- and phase-shift keying (APSK) constellation. It is a complex one-dimensional [constellation](/ref/Constellation) obtained by the union of $K$ component [PSK constellations](/ref/PSKConstellation), called *rings*.

    Parameters:
        orders: A $K$-tuple with the orders $M_k$ of each ring, for $k \in [0 : K)$.

        amplitudes: A $K$-tuple with the amplitudes $A_k$ of each ring, for $k \in [0 : K)$.

        phase_offsets: A $K$-tuple with the phase offsets $\phi_k$ of each ring, for $k \in [0 : K)$. If specified as a single float $\phi$, then it is assumed that $\phi_k = \phi$ for all $k \in [0 : K)$. The default value is `0.0`.

    Examples:
        1. The $8$-APSK constellation with $(M_0, M_1) = (4, 4)$, $(A_0, A_1) = (1, 2)$, and $(\phi_0, \phi_1) = (0, 0)$ is depicted below.
            <figure markdown>
            ![(4,4)-APSK constellation.](/fig/constellation_apsk_4_4.svg)
            </figure>

                >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))

        1. The $16$-APSK constellation with $(M_0, M_1) = (8, 8)$, $(A_0, A_1) = (1, 2)$, and $(\phi_0, \phi_1) = (0, 1/16)$ is depicted below.
            <figure markdown>
            ![(8,8)-APSK constellation.](/fig/constellation_apsk_8_8.svg)
            </figure>

                >>> const = komm.APSKConstellation(
                ...     orders=(8, 8),
                ...     amplitudes=(1.0, 2.0),
                ...     phase_offsets=(0.0, 1 / 16)
                ... )

        1. The $16$-APSK constellation with $(M_0, M_1) = (4, 12)$, $(A_0, A_1) = (\sqrt{2}, 3)$, and $(\phi_0, \phi_1) = (1/8, 0)$ is depicted below.
            <figure markdown>
            ![(4,12)-APSK constellation.](/fig/constellation_apsk_4_12.svg)
            </figure>

                >>> const = komm.APSKConstellation(
                ...     orders=(4, 12),
                ...     amplitudes=(np.sqrt(2), 3.0),
                ...     phase_offsets=(1 / 8, 0.0)
                ... )
    """

    _orders: tuple[int, ...]
    _amplitudes: tuple[float, ...]
    _phase_offsets: tuple[float, ...]

    def __init__(
        self,
        orders: tuple[int, ...],
        amplitudes: tuple[float, ...],
        phase_offsets: float | tuple[float, ...] = 0.0,
    ) -> None:
        self._orders = orders
        self._amplitudes = amplitudes
        if isinstance(phase_offsets, tuple):
            self._phase_offsets = phase_offsets
        else:
            self._phase_offsets = (phase_offsets,) * len(orders)

    def __repr__(self) -> str:
        args = ", ".join([
            f"orders={tuple(map(int, self._orders))}",
            f"amplitudes={tuple(map(float, self._amplitudes))}",
            f"phase_offsets={tuple(map(float, self._phase_offsets))}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.matrix
            array([[ 1.+0.j],
                   [ 0.+1.j],
                   [-1.+0.j],
                   [ 0.-1.j],
                   [ 2.+0.j],
                   [ 0.+2.j],
                   [-2.+0.j],
                   [ 0.-2.j]])
        """
        Ms, As, φs = self._orders, self._amplitudes, self._phase_offsets
        matrices = [PSKConstellation(M, A, φ).matrix for M, A, φ in zip(Ms, As, φs)]
        matrix = np.concatenate(matrices)
        return matrix.reshape(-1, 1)

    @property
    def order(self) -> int:
        r"""
        For the APSK constellation, it is given by
        $$
            M = \sum_{k \in [0:K)} M_k.
        $$

        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.order
            8
        """
        return sum(self._orders)

    @property
    def dimension(self) -> int:
        r"""
        For the APSK constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.complexfloating]:
        r"""
        For uniform priors, the mean of the APSK constellation is given by
        $$
            \mathbf{m} = 0.
        $$

        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.mean()
            array([0.+0.j])
        """
        if priors is None:
            return np.zeros((1,), dtype=complex)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the APSK constellation is given by
        $$
            E = \frac{1}{M} \sum_{k \in [0:K)} A^2 M_k.
        $$

        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.mean_energy()
            np.float64(2.5)
        """
        if priors is None:
            Ms, As = self._orders, self._amplitudes
            return np.sum([M * A**2 for (M, A) in zip(Ms, As)]) / self.order
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.minimum_distance()
            np.float64(1.0)
        """
        return super().minimum_distance()

    def indices_to_symbols(
        self, indices: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.indices_to_symbols([3, 0])
            array([0.-1.j, 1.+0.j])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.closest_indices([0.1 - 1.1j, 1.2 + 0.1j])
            array([3, 0])
        """
        return super().closest_indices(received)

    def closest_symbols(
        self, received: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
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
            >>> const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> const.posteriors([0.1 - 1.1j], snr=2.0).round(3)
            array([0.104, 0.015, 0.076, 0.516, 0.011, 0.   , 0.006, 0.272])
        """
        return super().posteriors(received, snr, priors)
