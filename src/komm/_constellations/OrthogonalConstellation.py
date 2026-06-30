from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class OrthogonalConstellation(abc.Constellation[np.floating]):
    r"""
    Orthogonal constellation. It is a real $M$-dimensional [constellation](/ref/Constellation) of order $M$ whose symbols are mutually orthogonal and have the same energy. The $i$-th symbol is given by
    $$
        \mathbf{x}_i = A \mathbf{e}_i, \quad i \in [0 : M),
    $$
    where $A$ is the *amplitude*, and $\mathbf{e}_i$ is the $i$-th standard basis vector of $\mathbb{R}^M$. For more details, see <cite>PS08, Sec. 3.2–4</cite>.

    Parameters:
        order: The order $M$ of the constellation.

        amplitude: The amplitude $A$ of the constellation. The default value is `1.0`.

    Examples:
        1. The $4$-ary orthogonal constellation with amplitude $A = 1$ is given by

                >>> const = komm.OrthogonalConstellation(4)
                >>> const.matrix
                array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        1. The $2$-ary orthogonal constellation with amplitude $A = 3$ is given by

                >>> const = komm.OrthogonalConstellation(2, amplitude=3.0)
                >>> const.matrix
                array([[3., 0.],
                       [0., 3.]])
    """

    def __init__(self, order: int, amplitude: float = 1.0) -> None:
        self._order = order
        self._amplitude = np.float64(amplitude)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"amplitude={self._amplitude}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.floating]:
        r"""
        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.matrix
            array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
        """
        M, A = self._order, self._amplitude
        return A * np.eye(M)

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the orthogonal constellation, it is given by $N = M$.

        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.dimension
            4
        """
        return self._order

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.floating]:
        r"""
        For uniform priors, the mean of the orthogonal constellation is given by
        $$
            \mathbf{m} = \frac{A}{M} \mathbf{1}.
        $$

        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.mean()
            array([0.25, 0.25, 0.25, 0.25])
        """
        if priors is None:
            M, A = self._order, self._amplitude
            return np.full(M, A / M)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the orthogonal constellation is given by
        $$
            E = A^2.
        $$

        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.mean_energy()
            np.float64(1.0)
        """
        if priors is None:
            return self._amplitude**2
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the orthogonal constellation, the minimum distance is given by
        $$
            d_{\min} = A \sqrt{2}.
        $$

        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.minimum_distance()  # doctest: +FLOAT_CMP
            np.float64(1.4142135623730951)
        """
        return self._amplitude * np.sqrt(2)

    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.indices_to_symbols([1, 3])
            array([0., 1., 0., 0., 0., 0., 0., 1.])
            >>> const.indices_to_symbols([[1, 3], [0, 2]])
            array([[0., 1., 0., 0., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0., 0., 1., 0.]])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.closest_indices([0.2, -0.1, 0.9, 0.3, 1.2, 0.1, -0.3, 0.0])
            array([2, 0])
        """
        M = self._order
        received = np.asarray(received)
        indices = np.argmax(received.reshape(-1, M), axis=-1)
        return indices.reshape(*received.shape[:-1], -1)

    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.closest_symbols([0.2, -0.1, 0.9, 0.3, 1.2, 0.1, -0.3, 0.0])
            array([0., 0., 1., 0., 1., 0., 0., 0.])
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
            >>> const = komm.OrthogonalConstellation(4)
            >>> const.posteriors([0.2, -0.1, 0.9, 0.3], noise_power=0.5).round(3)
            array([0.147, 0.08 , 0.594, 0.179])
        """
        return super().posteriors(received, noise_power, priors)
