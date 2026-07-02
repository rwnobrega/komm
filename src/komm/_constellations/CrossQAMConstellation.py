from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D
from .QAMConstellation import QAMConstellation


class CrossQAMConstellation(abc.Constellation[np.complexfloating]):
    r"""
    Cross quadrature amplitude modulation (cross-QAM) constellation. It is a complex one-dimensional [constellation](/ref/Constellation) defined only for orders $M = 2^k$ with $k$ odd and $k \geq 5$ (that is, $M = 32, 128, 512, \ldots$), for which a square [QAM constellation](/ref/QAMConstellation) does not exist. More precisely, start with a smaller square QAM constellation with order $2^{k-1}$ and extend each side of the QAM square by adding $2^{k-3}$ points, ignoring the corners in this extension; this leaves a total of $2^{k-1} + 4 \cdot 2^{k-3} = 2^k$ points resulting in a cross shape. Equivalently, it may be obtained from a bigger $L \times L$ square QAM constellation, where $L = 3 \sqrt{2^{k-3}}$, by removing the four corner regions. For more details, see <cite>Hay04, Sec. 6.4</cite>.

    Parameters:
        order: The order $M$ of the constellation. Must be of the form $M = 2^k$ with $k$ odd and $k \geq 5$.

        delta: The distance $\delta$ between adjacent symbols (along the in-phase and quadrature axes). The default value is `2.0`.

        phase_offset: The phase offset $\phi$ of the constellation (in turns, not radians). The default value is `0.0`.

    Examples:
        The $32$-cross-QAM constellation with $\delta = 2$ is depicted below.
        <figure markdown>
        ![32-cross-QAM constellation.](/fig/constellation_cross_qam_32.svg)
        </figure>

            >>> const = komm.CrossQAMConstellation(32)
    """

    def __init__(
        self, order: int, delta: float = 2.0, phase_offset: float = 0.0
    ) -> None:
        k = order.bit_length() - 1
        if order < 32 or order & (order - 1) != 0 or k % 2 == 0:
            raise ValueError(
                "'order' must be of the form 2^k with k odd and k >= 5"
                " (e.g., 32, 128, 512)"
            )
        self._order = order
        self._delta = np.float64(delta)
        self._phase_offset = np.float64(phase_offset)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"delta={self._delta}",
            f"phase_offset={self._phase_offset}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.matrix
            array([[-5.-3.j],
                   [-5.-1.j],
                   [-5.+1.j],
                   [-5.+3.j],
                   [-3.-5.j],
                   [-3.-3.j],
                   [-3.-1.j],
                   [-3.+1.j],
                   [-3.+3.j],
                   [-3.+5.j],
                   [-1.-5.j],
                   [-1.-3.j],
                   [-1.-1.j],
                   [-1.+1.j],
                   [-1.+3.j],
                   [-1.+5.j],
                   [ 1.-5.j],
                   [ 1.-3.j],
                   [ 1.-1.j],
                   [ 1.+1.j],
                   [ 1.+3.j],
                   [ 1.+5.j],
                   [ 3.-5.j],
                   [ 3.-3.j],
                   [ 3.-1.j],
                   [ 3.+1.j],
                   [ 3.+3.j],
                   [ 3.+5.j],
                   [ 5.-3.j],
                   [ 5.-1.j],
                   [ 5.+1.j],
                   [ 5.+3.j]])
        """
        M, δ, φ = self._order, self._delta, self._phase_offset
        k = M.bit_length() - 1
        L = 3 * 2 ** ((k - 3) // 2)
        big = QAMConstellation(orders=(L, L), deltas=δ).matrix
        threshold = 2 ** ((k - 1) // 2) * (δ / 2.0)
        is_corner = (np.abs(big.real) > threshold) & (np.abs(big.imag) > threshold)
        matrix = big[~is_corner] * np.exp(2j * np.pi * φ)
        return matrix.reshape(-1, 1)

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.order
            32
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the cross-QAM constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.complexfloating]:
        r"""
        For uniform priors, the mean of the cross-QAM constellation is given by
        $$
            \mathbf{m} = 0.
        $$

        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.mean()
            array([0.+0.j])
        """
        if priors is None:
            return np.zeros((1,), dtype=complex)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the cross-QAM constellation is given by
        $$
            E = \frac{\delta^2}{192} (31 M - 32).
        $$

        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.mean_energy()
            np.float64(20.0)
        """
        if priors is None:
            M, δ = self._order, self._delta
            return np.float64(δ**2 * (31 * M - 32) / 192)
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the cross-QAM constellation, the minimum distance is given by
        $$
            d_{\min} = \delta.
        $$

        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.minimum_distance()
            np.float64(2.0)
        """
        return self._delta

    def indices_to_symbols(
        self, indices: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.indices_to_symbols([12, 0])
            array([-1.-1.j, -5.-3.j])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.closest_indices([-1.1 - 0.9j, 5.2 + 5.1j])
            array([12, 31])
        """
        return super().closest_indices(received)

    def closest_symbols(
        self, received: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.closest_symbols([-1.1 - 0.9j, 5.2 + 5.1j])
            array([-1.-1.j,  5.+3.j])
        """
        return super().closest_symbols(received)

    def posteriors(
        self,
        received: npt.ArrayLike,
        noise_power: float,
        priors: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.CrossQAMConstellation(32)
            >>> const.posteriors([-1.1 - 0.9j], noise_power=2.0).round(3)
            array([0.   , 0.   , 0.   , 0.   , 0.   , 0.011, 0.101, 0.017, 0.   ,
                   0.   , 0.   , 0.068, 0.613, 0.101, 0.   , 0.   , 0.   , 0.008,
                   0.068, 0.011, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
                   0.   , 0.   , 0.   , 0.   , 0.   ])
        """
        return super().posteriors(received, noise_power, priors)
