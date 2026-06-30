from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class SimplexConstellation(abc.Constellation[np.floating]):
    r"""
    Simplex constellation. It is a real $M$-dimensional [constellation](/ref/Constellation) of order $M$ obtained from an [orthogonal constellation](/ref/OrthogonalConstellation) by subtracting from each symbol the mean of all the symbols. The $i$-th symbol is given by
    $$
        \mathbf{x}_i = A \left( \mathbf{e}_i - \frac{1}{M} \mathbf{1} \right), \quad i \in [0 : M),
    $$
    where $A$ is the *base amplitude*, and $\mathbf{e}_i$ is the $i$-th standard basis vector of $\mathbb{R}^M$. The symbols are equidistant, equicorrelated, and lie in an $(M-1)$-dimensional subspace. The simplex constellation achieves the same minimum distance as the orthogonal constellation, but with smaller energy. For more details, see <cite>PS08, Sec. 3.2–4</cite>.

    Parameters:
        order: The order $M$ of the constellation.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

    Examples:
        1. The $4$-ary simplex constellation with base amplitude $A = 1$ is given by

                >>> const = komm.SimplexConstellation(4)
                >>> const.matrix
                array([[ 0.75, -0.25, -0.25, -0.25],
                       [-0.25,  0.75, -0.25, -0.25],
                       [-0.25, -0.25,  0.75, -0.25],
                       [-0.25, -0.25, -0.25,  0.75]])

        1. The $2$-ary simplex constellation with base amplitude $A = 1$ is given by

                >>> const = komm.SimplexConstellation(2)
                >>> const.matrix
                array([[ 0.5, -0.5],
                       [-0.5,  0.5]])
    """

    def __init__(self, order: int, base_amplitude: float = 1.0) -> None:
        self._order = order
        self._base_amplitude = np.float64(base_amplitude)

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"base_amplitude={self._base_amplitude}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @cache
    def matrix(self) -> Array2D[np.floating]:
        r"""
        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.matrix
            array([[ 0.75, -0.25, -0.25, -0.25],
                   [-0.25,  0.75, -0.25, -0.25],
                   [-0.25, -0.25,  0.75, -0.25],
                   [-0.25, -0.25, -0.25,  0.75]])
        """
        M, A = self._order, self._base_amplitude
        return A * (np.eye(M) - np.ones((M, M)) / M)

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the simplex constellation, it is given by $N = M$. Note, however, that the symbols lie in an $(M-1)$-dimensional subspace.

        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.dimension
            4
        """
        return self._order

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.floating]:
        r"""
        For uniform priors, the mean of the simplex constellation is given by
        $$
            \mathbf{m} = \mathbf{0}.
        $$

        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.mean()
            array([0., 0., 0., 0.])
        """
        if priors is None:
            return np.zeros(self._order)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the simplex constellation is given by
        $$
            E = A^2 \left( 1 - \frac{1}{M} \right).
        $$

        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.mean_energy()
            np.float64(0.75)
        """
        if priors is None:
            M, A = self._order, self._base_amplitude
            return A**2 * (M - 1) / M
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the simplex constellation, the minimum distance is given by
        $$
            d_{\min} = A \sqrt{2}.
        $$

        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.minimum_distance()  # doctest: +FLOAT_CMP
            np.float64(1.4142135623730951)
        """
        A = self._base_amplitude
        return A * np.sqrt(2)

    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.indices_to_symbols([1, 3])
            array([-0.25,  0.75, -0.25, -0.25, -0.25, -0.25, -0.25,  0.75])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.closest_indices([0.6, -0.1, -0.2, -0.3])
            array([0])
        """
        M = self._order
        received = np.asarray(received)
        indices = np.argmax(received.reshape(-1, M), axis=-1)
        return indices.reshape(*received.shape[:-1], -1)

    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.SimplexConstellation(4)
            >>> const.closest_symbols([0.6, -0.1, -0.2, -0.3])
            array([ 0.75, -0.25, -0.25, -0.25])
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
            >>> const = komm.SimplexConstellation(4)
            >>> const.posteriors([0.6, -0.1, -0.2, -0.3], noise_power=0.5).round(3)
            array([0.62 , 0.153, 0.125, 0.102])
        """
        return super().posteriors(received, noise_power, priors)
