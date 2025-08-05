from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D, Array2D

T = TypeVar("T", np.floating, np.complexfloating)


class Constellation(abc.Constellation[T]):
    r"""
    General real or complex constellation. A *constellation* of *dimension* $N$ and *order* $M$ is defined by an ordered set $\\{ \mathbf{x}_i : i \in [0:M) \\}$ of $M$ distinct points in $\mathbb{R}^N$ or $\mathbb{C}^N$, called *symbols*. In this class, the constellation is represented by a matrix $\mathbf{X} \in \mathbb{R}^{M \times N}$ or $\mathbf{X} \in \mathbb{C}^{M \times N}$, where the $i$-th row of $\mathbf{X}$ corresponds to symbol $\mathbf{x}_i$. For more details, see <cite>SA15, Sec. 2.5.1</cite>.

    Parameters:
        matrix: The constellation matrix $\mathbf{X}$. Must be a 2D-array of shape $(M, N)$ with real or complex entries.

    Examples:
        The real constellation depicted in the figure below has $M = 5$ and $N = 2$.
        <figure markdown>
        ![Example for real constellation with M = 5 and N = 2](/fig/constellation_crux.svg)
        </figure>

            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
    """

    def __init__(self, matrix: npt.ArrayLike) -> None:
        dtype = complex if np.iscomplexobj(matrix) else float
        matrix = np.asarray(matrix, dtype=dtype)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError("'matrix' must be a 2D-array")
        self._matrix = matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix.tolist()})"

    @property
    def matrix(self) -> Array2D[T]:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.matrix
            array([[ 0.,  4.],
                   [-2.,  2.],
                   [ 2.,  2.],
                   [ 1.,  1.],
                   [ 0., -2.]])
        """
        return self._matrix

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.order
            5
        """
        return super().order

    @property
    def dimension(self) -> int:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.dimension
            2
        """
        return super().dimension

    def mean(self, priors: npt.ArrayLike | None = None) -> Array1D[T]:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.mean()
            array([0.2, 1.4])
        """
        return super().mean(priors)

    def mean_energy(self, priors: npt.ArrayLike | None = None) -> np.floating:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.mean_energy()  # doctest: +FLOAT_CMP
            np.float64(7.6)
        """
        return super().mean_energy(priors)

    def minimum_distance(self) -> np.floating:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.minimum_distance()  # doctest: +FLOAT_CMP
            np.float64(1.4142135623730951)
        """
        return super().minimum_distance()

    def indices_to_symbols(self, indices: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.indices_to_symbols([3, 0])
            array([1., 1., 0., 4.])
            >>> const.indices_to_symbols([[3, 0], [1, 2]])
            array([[ 1.,  1.,  0.,  4.],
                   [-2.,  2.,  2.,  2.]])
        """
        return super().indices_to_symbols(indices)

    def closest_indices(self, received: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.closest_indices([0.3, 1.8, 0.0, 5.0])
            array([3, 0])
            >>> const.closest_indices([[0.3, 1.8], [0.0, 5.0]])
            array([[3],
                   [0]])
        """
        return super().closest_indices(received)

    def closest_symbols(self, received: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Examples:
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.closest_symbols([0.3, 1.8, 0.0, 5.0])
            array([1., 1., 0., 4.])
            >>> const.closest_symbols([[0.3, 1.8], [0.0, 5.0]])
            array([[1., 1.], [0., 4.]])
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
            >>> const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
            >>> const.posteriors([0.3, 1.8, 0.0, 5.0], snr=2.0).round(4)
            array([0.1565, 0.1408, 0.2649, 0.4253, 0.0125,
                   0.9092, 0.0387, 0.0387, 0.0135, 0.    ])
            >>> const.posteriors([[0.3, 1.8], [0.0, 5.0]], snr=2.0).round(4)
            array([[0.1565, 0.1408, 0.2649, 0.4253, 0.0125],
                   [0.9092, 0.0387, 0.0387, 0.0135, 0.    ]])
        """
        return super().posteriors(received, snr, priors)
