from collections.abc import Callable
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc


class ScalarQuantizer(abc.ScalarQuantizer):
    r"""
    General scalar quantizer. It is defined by a list of *levels*, $y_0, y_1, \ldots, y_{L-1}$, and a list of *thresholds*, $\lambda_0, \lambda_1, \ldots, \lambda_L$, satisfying
    $$
        -\infty = \lambda_0 < y_0 < \lambda_1 < y_1 < \cdots < \lambda_{L - 1} < y_{L - 1} < \lambda_L = +\infty.
    $$
    Given an input $x \in \mathbb{R}$, the output of the quantizer is given by $y = y_i$ if and only if $\lambda_i \leq x < \lambda_{i+1}$, where $i \in [0:L)$. For more details, see <cite>Say06, Ch. 9</cite>.

    Parameters:
        levels: The quantizer levels $y_0, y_1, \ldots, y_{L-1}$. It should be a list floats of length $L$.

        thresholds: The quantizer finite thresholds $\lambda_1, \lambda_2, \ldots, \lambda_{L-1}$. It should be a list of floats of length $L - 1$.

    Examples:
        The $5$-level scalar quantizer whose characteristic (input × output) curve is depicted in the figure below has levels
        $$
            y_0 = -2, ~ y_1 = -1, ~ y_2 = 0, ~ y_3 = 1, ~ y_4 = 2,
        $$
        and thresholds
        $$
            \lambda_0 = -\infty, ~ \lambda_1 = -1.5, ~ \lambda_2 = -0.3, ~ \lambda_3 = 0.8, ~ \lambda_4 = 1.4, ~ \lambda_5 = \infty.
        $$

        <figure markdown>
          ![Scalar quantizer example.](/fig/scalar_quantizer_5.svg)
        </figure>

        >>> quantizer = komm.ScalarQuantizer(
        ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
        ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
        ... )
    """

    def __init__(self, levels: npt.ArrayLike, thresholds: npt.ArrayLike) -> None:
        self._levels = np.asarray(levels, dtype=float)
        self._thresholds = np.asarray(thresholds, dtype=float)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.thresholds.size == self.levels.size - 1:
            raise ValueError("'len(thresholds)' must be equal to 'len(levels) - 1'")

        interleaved = np.empty(2 * self.levels.size - 1, dtype=float)
        interleaved[0::2] = self.levels
        interleaved[1::2] = self.thresholds

        if not np.array_equal(np.unique(interleaved), interleaved):
            raise ValueError("invalid values for 'levels' and 'thresholds'")

    def __repr__(self) -> str:
        args = ", ".join([
            f"levels={self.levels.tolist()}",
            f"thresholds={self.thresholds.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.levels
            array([-2., -1.,  0.,  1.,  2.])
        """
        return self._levels

    @cached_property
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.thresholds
            array([-1.5, -0.3,  0.8,  1.4])
        """
        return self._thresholds

    def mean_squared_error(
        self,
        input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        input_range: tuple[float, float],
        points_per_interval: int = 4096,
    ) -> float:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer.mean_squared_error(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ... )  # doctest: +FLOAT_CMP
            0.13598089455499335
        """
        return super().mean_squared_error(input_pdf, input_range, points_per_interval)

    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.digitize([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
            array([0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 4])
        """
        return super().digitize(input)

    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.quantize([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
            array([-2., -2., -1., -1., -1.,  0.,  0.,  1.,  2.,  2.,  2.])
        """
        return super().quantize(input)
