from collections.abc import Iterable
from functools import cache
from math import isqrt

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D
from .PAMConstellation import PAMConstellation


class QAMConstellation(abc.Constellation[np.complexfloating]):
    r"""
    Quadrature amplitude modulation (QAM) constellation. It is a complex one-dimensional [constellation](/ref/Constellation) obtained by a Cartesian product of two [PAM constellations](/ref/PAMConstellation), namely, the *in-phase constellation*, and the *quadrature constellation*. For more details, see <cite>SA15, Sec. 2.5.1</cite>.

    Parameters:
        orders: A tuple $(M_\mathrm{I}, M_\mathrm{Q})$ with the orders of the in-phase and quadrature constellations, respectively. If specified as a single integer $M$, then it is assumed that $M_\mathrm{I} = M_\mathrm{Q} = \sqrt{M}$; in this case, $M$ must be a perfect square.

        deltas: A tuple $(\Delta_\mathrm{I}, \Delta_\mathrm{Q})$ with the distances of the in-phase and quadrature constellations, respectively. If specified as a single float $\Delta$, then it is assumed that $\Delta_\mathrm{I} = \Delta_\mathrm{Q} = \Delta$. The default value is `2.0`.

        phase_offset: The phase offset $\phi$ of the constellation (in turns, not radians). The default value is `0.0`.

    Examples:
        1. The square $16$-QAM constellation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ and $(\Delta_\mathrm{I}, \Delta_\mathrm{Q}) = (2, 2)$ is depicted below.
            <figure markdown>
            ![16-QAM constellation.](/fig/constellation_qam_16.svg)
            </figure>

                >>> const = komm.QAMConstellation(16)

        1.  The rectangular $8$-QAM constellation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 2)$ and $(\Delta_\mathrm{I}, \Delta_\mathrm{Q}) = (2, 4)$ is depicted below.
            <figure markdown>
            ![8-QAM constellation.](/fig/constellation_qam_8.svg)
            </figure>

                >>> const = komm.QAMConstellation(orders=(4, 2), deltas=(2.0, 4.0))
    """

    _orders: tuple[int, int]
    _deltas: tuple[float, float]
    phase_offset: float

    def __init__(
        self,
        orders: tuple[int, int] | int,
        deltas: tuple[float, float] | float = 2.0,
        phase_offset: float = 0.0,
    ) -> None:
        if isinstance(orders, Iterable):
            self._orders = orders
        else:
            if not isqrt(orders) ** 2 == orders:
                raise ValueError(
                    "when a single integer, 'orders' must be a perfect square"
                )
            self._orders = (isqrt(orders), isqrt(orders))
        if isinstance(deltas, Iterable):
            self._deltas = deltas
        else:
            self._deltas = (deltas, deltas)
        self._phase_offset = phase_offset

    def __repr__(self) -> str:
        args = ", ".join([
            f"orders={tuple(map(int, self._orders))}",
            f"deltas={tuple(map(float, self._deltas))}",
            f"phase_offset={self._phase_offset}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.matrix
            array([[-3.-3.j],
                   [-3.-1.j],
                   [-3.+1.j],
                   [-3.+3.j],
                   [-1.-3.j],
                   [-1.-1.j],
                   [-1.+1.j],
                   [-1.+3.j],
                   [ 1.-3.j],
                   [ 1.-1.j],
                   [ 1.+1.j],
                   [ 1.+3.j],
                   [ 3.-3.j],
                   [ 3.-1.j],
                   [ 3.+1.j],
                   [ 3.+3.j]])
        """
        Mi, Mq = self._orders
        Δi, Δq = self._deltas
        φ = self._phase_offset
        matrix_i = PAMConstellation(Mi, Δi).matrix
        matrix_q = PAMConstellation(Mq, Δq).matrix
        matrix = (matrix_i + 1j * matrix_q.T) * np.exp(2j * np.pi * φ)
        return matrix.reshape(-1, 1)

    @property
    def order(self) -> int:
        r"""
        For the QAM constellation, it is given by
        $$
            M = M_\mathrm{I} M_\mathrm{Q}.
        $$

        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.order
            16
        """
        Mi, Mq = self._orders
        return Mi * Mq

    @property
    def dimension(self) -> int:
        r"""
        For the QAM constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.complexfloating]:
        r"""
        For uniform priors, the mean of the QAM constellation is given by
        $$
            \mathbf{m} = 0.
        $$

        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.mean()
            array([0.+0.j])
        """
        if priors is None:
            return np.zeros((1,), dtype=complex)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the QAM constellation is given by
        $$
            E = \frac{\Delta_\mathrm{I}^2}{12}(M_\mathrm{I}^2 - 1) + \frac{\Delta_\mathrm{Q}^2}{12}(M_\mathrm{Q}^2 - 1).
        $$
        For the special case of a square QAM constellation, this simplifies to
        $$
            E = \frac{\Delta^2}{6}(M - 1).
        $$

        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.mean_energy()
            np.float64(10.0)
        """
        if priors is None:
            Mi, Mq = self._orders
            Δi, Δq = self._deltas
            return np.float64(Δi**2 * (Mi**2 - 1) / 12 + Δq**2 * (Mq**2 - 1) / 12)
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the QAM constellation, the minimum distance is given by
        $$
            d_{\min} = \min(\Delta_\mathrm{I}, \Delta_\mathrm{Q}).
        $$

        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.minimum_distance()
            np.float64(2.0)
        """
        return np.min(self._deltas)

    def indices_to_symbols(
        self, indices: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.indices_to_symbols([3, 0])
            array([-3.+3.j, -3.-3.j])
            >>> const.indices_to_symbols([[3, 0], [1, 2]])
            array([[-3.+3.j, -3.-3.j],
                   [-3.-1.j, -3.+1.j]])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.closest_indices([-3.1 + 2.9j, -3 - 3.5j])
            array([3, 0])
        """
        Mi, Mq = self._orders
        Δi, Δq = self._deltas
        φ = self._phase_offset
        Ai, Aq = Δi / 2, Δq / 2
        received_i = np.real(np.multiply(received, np.exp(-2j * np.pi * φ))) / Ai
        received_q = np.imag(np.multiply(received, np.exp(-2j * np.pi * φ))) / Aq
        indices_i = np.clip(np.around((received_i + Mi - 1) / 2), 0, Mi - 1).astype(int)
        indices_q = np.clip(np.around((received_q + Mq - 1) / 2), 0, Mq - 1).astype(int)
        indices = Mq * indices_i + indices_q
        return indices

    def closest_symbols(
        self, received: npt.ArrayLike
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> const = komm.QAMConstellation(16)
            >>> const.closest_symbols([-3.1 + 2.9j, -3 - 3.5j])
            array([-3.+3.j, -3.-3.j])
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
            >>> const = komm.QAMConstellation(16)
            >>> const.posteriors([-3.1 + 2.9j], snr=2.0).round(3)
            array([0.   , 0.021, 0.219, 0.449, 0.   , 0.009, 0.091, 0.186, 0.   ,
                   0.001, 0.008, 0.016, 0.   , 0.   , 0.   , 0.   ])
        """
        return super().posteriors(received, snr, priors)
