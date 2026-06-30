from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D


class BiorthogonalConstellation(abc.Constellation[np.floating]):
    r"""
    Biorthogonal constellation. It is a real $(M/2)$-dimensional [constellation](/ref/Constellation) of order $M$ (which must be even) obtained from the $M/2$ symbols of an [orthogonal constellation](/ref/OrthogonalConstellation) together with their negatives. The $i$-th symbol is given by
    $$
        \mathbf{x}_i = \begin{cases}
            A \mathbf{e}_i, & i \in [0 : M/2), \\\\
            -A \mathbf{e}\_{i - M/2}, & i \in [M/2 : M),
        \end{cases}
    $$
    where $A$ is the *amplitude*, and $\mathbf{e}_i$ is the $i$-th standard basis vector of $\mathbb{R}^{M/2}$. For more details, see <cite>PS08, Sec. 3.2–4</cite>.

    Parameters:
        order: The order $M$ of the constellation. Must be an even integer.

        amplitude: The amplitude $A$ of the constellation. The default value is `1.0`.

    Examples:
        1. The $4$-ary biorthogonal constellation with amplitude $A = 1$ is given by

                >>> const = komm.BiorthogonalConstellation(4)
                >>> const.matrix
                array([[ 1.,  0.],
                       [ 0.,  1.],
                       [-1.,  0.],
                       [ 0., -1.]])

        1. The $2$-ary biorthogonal constellation with amplitude $A = 3$ is given by

                >>> const = komm.BiorthogonalConstellation(2, amplitude=3.0)
                >>> const.matrix
                array([[ 3.],
                       [-3.]])
    """

    def __init__(self, order: int, amplitude: float = 1.0) -> None:
        if order % 2 != 0:
            raise ValueError(f"'order' must be even (got {order})")
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
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.matrix
            array([[ 1.,  0.],
                   [ 0.,  1.],
                   [-1.,  0.],
                   [ 0., -1.]])
        """
        M, A = self._order, self._amplitude
        matrix = np.vstack([A * np.eye(M // 2), -A * np.eye(M // 2)])
        matrix += 0.0  # to avoid -0.0
        return matrix

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.order
            4
        """
        return self._order

    @property
    def dimension(self) -> int:
        r"""
        For the biorthogonal constellation, it is given by $N = M/2$.

        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.dimension
            2
        """
        return self._order // 2

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[np.floating]:
        r"""
        For uniform priors, the mean of the biorthogonal constellation is given by
        $$
            \mathbf{m} = \mathbf{0}.
        $$

        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.mean()
            array([0., 0.])
        """
        if priors is None:
            return np.zeros(self._order // 2)
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        For uniform priors, the mean energy of the biorthogonal constellation is given by
        $$
            E = A^2.
        $$

        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.mean_energy()
            np.float64(1.0)
        """
        if priors is None:
            return self._amplitude**2
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        For the biorthogonal constellation, the minimum distance is given by
        $$
            d_{\min} = \begin{cases}
                2A, & M = 2, \\\\
                A \sqrt{2}, & M \geq 4.
            \end{cases}
        $$

        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.minimum_distance()  # doctest: +FLOAT_CMP
            np.float64(1.4142135623730951)

            >>> const = komm.BiorthogonalConstellation(2)
            >>> const.minimum_distance()
            np.float64(2.0)
        """
        A = self._amplitude
        return 2 * A if self._order == 2 else A * np.sqrt(2)

    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.indices_to_symbols([0, 3])
            array([ 1.,  0.,  0., -1.])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.closest_indices([0.7, 0.2, -0.1, -0.9])
            array([0, 3])
        """
        M = self._order
        N = M // 2
        received = np.asarray(received)
        r = received.reshape(-1, N)
        scores = np.concatenate([r, -r], axis=-1)
        indices = np.argmax(scores, axis=-1)
        return indices.reshape(*received.shape[:-1], -1)

    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.closest_symbols([0.7, 0.2, -0.1, -0.9])
            array([ 1.,  0.,  0., -1.])
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
            >>> const = komm.BiorthogonalConstellation(4)
            >>> const.posteriors([0.7, 0.2], noise_power=0.5).round(3)
            array([0.627, 0.231, 0.038, 0.104])
        """
        return super().posteriors(received, noise_power, priors)
