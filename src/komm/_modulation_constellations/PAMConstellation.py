from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class PAMConstellation(abc.Constellation[np.floating]):
    r"""
    Pulse-amplitude modulation (PAM) constellation. It is a real one-dimensional [constellation](/ref/Constellation) in which the symbols are *uniformly arranged* in the real line and centered about the origin. For more details, see <cite>SA15, Sec. 2.5.1</cite>.

    Parameters:
        order: The order $M$ of the constellation.

        delta: The distance $\Delta$ between adjacent symbols. The default value is `2.0`.

    Examples:
        1. The $4$-PAM constellation with $\Delta = 2$ is depicted below.
            <figure markdown>
            ![4-PAM constellation.](/fig/constellation_pam_4.svg)
            </figure>

                >>> const = komm.PAMConstellation(4)

        1. The $7$-PAM constellation with $\Delta = 5$ is depicted below.
            <figure markdown>
            ![7-PAM constellation.](/fig/constellation_pam_7.svg)
            </figure>

                >>> const = komm.PAMConstellation(7, delta=5)
    """

    def __init__(self, order: int, delta: float = 2.0) -> None:
        self._order = order
        self._delta = np.float64(delta)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"delta={self._delta}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.floating]:
        r"""
        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.matrix
            array([[-3.],
                   [-1.],
                   [ 1.],
                   [ 3.]])
        """
        M, Δ = self._order, self._delta
        peak = (M - 1) * Δ / 2
        return np.linspace(-peak, peak, num=M, endpoint=True).reshape(-1, 1)

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the PAM constellation, it is given by $N = 1$.

        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.dimension
            1
        """
        return 1

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.floating]:
        r"""
        For uniform priors, the mean of the PAM constellation is given by
        $$
            \mathbf{m} = 0.
        $$

        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.mean()
            array([0.])
        """
        if priors is None:
            return np.zeros((1,), dtype=float)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the PAM constellation is given by
        $$
            E = \frac{\Delta^2}{12}(M^2 - 1).
        $$

        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.mean_energy()
            np.float64(5.0)
        """
        if priors is None:
            M, Δ = self._order, self._delta
            return np.float64(Δ**2 * (M**2 - 1) / 12)
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the PAM constellation, the minimum distance is given by
        $$
            d_{\min} = \Delta.
        $$

        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.minimum_distance()
            np.float64(2.0)
        """
        return self._delta

    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.indices_to_symbols([3, 0])
            array([ 3., -3.])
            >>> const.indices_to_symbols([[3, 0], [1, 2]])
            array([[ 3., -3.],
                   [-1.,  1.]])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.closest_indices([-0.8, 2.4])
            array([1, 3])
            >>> const.closest_indices([[-0.8, 2.4], [0.0, 10.0]])
            array([[1, 3],
                   [2, 3]])
        """
        M = self._order
        normalized = np.asarray(received, dtype=float) / self._delta
        indices = np.round(normalized + (M - 1) / 2).astype(int)
        indices = np.clip(indices, 0, M - 1)
        return indices

    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.PAMConstellation(4)
            >>> const.closest_symbols([-0.8, 2.4])
            array([-1.,  3.])
            >>> const.closest_symbols([[-0.8, 2.4], [0.0, 10.0]])
            array([[-1.,  3.],
                   [ 1.,  3.]])
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
            >>> const = komm.PAMConstellation(4)
            >>> const.posteriors([-0.8, 2.4], snr=2.0).round(3)
            array([0.103, 0.7  , 0.195, 0.002, 0.   , 0.007, 0.343, 0.65 ])
            >>> const.posteriors([[-0.8, 2.4], [0.0, 10.0]], snr=2.0).round(3)
            array([[0.103, 0.7  , 0.195, 0.002, 0.   , 0.007, 0.343, 0.65 ],
                   [0.02 , 0.48 , 0.48 , 0.02 , 0.   , 0.   , 0.   , 1.   ]])
        """
        return super().posteriors(received, snr, priors)
