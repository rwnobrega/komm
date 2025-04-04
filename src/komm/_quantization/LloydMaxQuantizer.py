from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class LloydMaxQuantizer(base.ScalarQuantizer):
    r"""
    Lloyd–Max scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) that minimizes the mean squared error (MSE) between the input signal $X$ and its quantized version. For more details, see <cite>Say06, Sec. 9.6.1</cite>.

    Parameters:
        input_pdf: The probability density function $f_X(x)$ of the input signal.

        input_range: The range $(x_\mathrm{min}, x_\mathrm{max})$ of the input signal.

        num_levels: The number $L$ of quantization levels. It must be greater than $1$.

    Examples:
        >>> uniform_pdf = lambda x: 1/8 * (np.abs(x) <= 4)
        >>> quantizer = komm.LloydMaxQuantizer(
        ...     input_pdf=uniform_pdf,
        ...     input_range=(-4, 4),
        ...     num_levels=8,
        ... )
        >>> quantizer.levels
        array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
        >>> quantizer.thresholds
        array([-3., -2., -1.,  0.,  1.,  2.,  3.])

        >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
        >>> quantizer = komm.LloydMaxQuantizer(
        ...     input_pdf=gaussian_pdf,
        ...     input_range=(-5, 5),
        ...     num_levels=8,
        ... )
        >>> quantizer.levels.round(3)
        array([-2.152, -1.344, -0.756, -0.245,  0.245,  0.756,  1.344,  2.152])
        >>> quantizer.thresholds.round(3)  # doctest: +FLOAT_CMP
        array([-1.748, -1.05 , -0.501,  0.   ,  0.501,  1.05 ,  1.748])
    """

    input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
    input_range: tuple[float, float]
    num_levels: int

    def __post_init__(self) -> None:
        self._levels, self._thresholds = lloyd_max_quantizer(
            self.input_pdf,
            self.num_levels,
            self.input_range,
            points_per_interval=4096,
            max_iter=1000,
        )

    @cached_property
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer = komm.LloydMaxQuantizer(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ...     num_levels=8,
            ... )
            >>> quantizer.levels.round(3)
            array([-2.152, -1.344, -0.756, -0.245,  0.245,  0.756,  1.344,  2.152])
        """
        return self._levels

    @cached_property
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer = komm.LloydMaxQuantizer(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ...     num_levels=8,
            ... )
            >>> quantizer.thresholds.round(3)  # doctest: +FLOAT_CMP
            array([-1.748, -1.05 , -0.501,  0.   ,  0.501,  1.05 ,  1.748])
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
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer = komm.LloydMaxQuantizer(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ...     num_levels=8,
            ... )
            >>> quantizer.mean_squared_error(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ... )  # doctest: +FLOAT_CMP
            0.034542475663845607
        """
        return super().mean_squared_error(input_pdf, input_range, points_per_interval)

    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer = komm.LloydMaxQuantizer(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ...     num_levels=8,
            ... )
            >>> quantizer.digitize([0, 1, 2, 3, 4, 5, 6, 7])
            array([4, 5, 7, 7, 7, 7, 7, 7])
        """
        return super().digitize(input)

    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer = komm.LloydMaxQuantizer(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ...     num_levels=8,
            ... )
            >>> quantizer.quantize([0, 1, 2, 3, 4, 5, 6, 7]).round(3)
            array([0.245, 0.756, 2.152, 2.152, 2.152, 2.152, 2.152, 2.152])
        """
        return super().quantize(input)


def lloyd_max_quantizer(
    input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    num_levels: int,
    input_range: tuple[float, float],
    points_per_interval: int,
    max_iter: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # See [Say06, eqs. (9.27) and (9.28)].
    x_min, x_max = input_range
    delta = (x_max - x_min) / num_levels

    # Initial guess
    levels = np.linspace(x_min + delta / 2, x_max - delta / 2, num=num_levels)
    thresholds = np.empty(num_levels + 1, dtype=float)
    new_levels = np.empty_like(levels)

    for _ in range(max_iter):
        thresholds[0] = x_min
        thresholds[1:-1] = 0.5 * (levels[:-1] + levels[1:])
        thresholds[-1] = x_max

        for i in range(num_levels):
            left, right = thresholds[i], thresholds[i + 1]
            x = np.linspace(left, right, num=points_per_interval, dtype=float)
            pdf = input_pdf(x)
            numerator = np.trapezoid(x * pdf, x)
            denominator = np.trapezoid(pdf, x)
            if denominator != 0:
                new_levels[i] = numerator / denominator
            else:  # Keep old level
                new_levels[i] = levels[i]
        if np.allclose(levels, new_levels):
            break
        levels = new_levels.copy()

    return new_levels, thresholds[1:-1]
