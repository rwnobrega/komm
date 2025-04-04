from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base


@dataclass(eq=False)
class UniformQuantizer(base.ScalarQuantizer):
    r"""
    Uniform scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) in which the separation between levels is constant, $\Delta$, and the thresholds are the mid-point between adjacent levels. For more details, see <cite>Say06, Sec. 9.4</cite>.

    Attributes:
        num_levels: The number of quantization levels $L$. It must be greater than $1$.

        input_range: The range $(x_\mathrm{min}, x_\mathrm{max})$ of the input signal. The default is $(-1.0, 1.0)$.

        choice: The choice for the uniform quantizer. Must be either `'mid-riser'` or `'mid-tread'`. The default is `'mid-riser'`.

    Examples:
        >>> quantizer = komm.UniformQuantizer(
        ...     num_levels=4,
        ...     input_range=(-1.0, 1.0),
        ...     choice='mid-riser',
        ... )
        >>> quantizer.levels
        array([-0.75, -0.25,  0.25,  0.75])
        >>> quantizer.thresholds
        array([-0.5,  0. ,  0.5])

        >>> quantizer = komm.UniformQuantizer(
        ...     num_levels=4,
        ...     input_range=(-1.0, 1.0),
        ...     choice='mid-tread',
        ... )
        >>> quantizer.levels
        array([-1. , -0.5,  0. ,  0.5])
        >>> quantizer.thresholds
        array([-0.75, -0.25,  0.25])

        >>> quantizer = komm.UniformQuantizer(
        ...     num_levels=4,
        ...     input_range=(0.0, 1.0),
        ...     choice='mid-tread',
        ... )
        >>> quantizer.levels
        array([0.  , 0.25, 0.5 , 0.75])
        >>> quantizer.thresholds
        array([0.125, 0.375, 0.625])
    """

    num_levels: int
    input_range: tuple[float, float] = field(default=(-1.0, 1.0))
    choice: Literal["mid-riser", "mid-tread"] = field(default="mid-riser")

    def __post_init__(self) -> None:
        if not self.num_levels > 1:
            raise ValueError("'num_levels' must be greater than 1")
        if not self.choice in ["mid-riser", "mid-tread"]:
            raise ValueError("'choice' must be either 'mid-riser' or 'mid-tread'")

    @cached_property
    def quantization_step(self) -> float:
        r"""
        The quantization step $\Delta$.

        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> quantizer.quantization_step
            2.0
        """
        x_min, x_max = self.input_range
        delta = (x_max - x_min) / self.num_levels
        return delta

    @cached_property
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> quantizer.levels
            array([-3., -1.,  1.,  3.])
        """
        num = self.num_levels
        x_min, x_max = self.input_range
        delta = (x_max - x_min) / num
        endpoint = (num % 2 == 0) if self.choice == "mid-riser" else (num % 2 == 1)
        min_level = x_min + (delta / 2) * endpoint
        max_level = x_max - (delta / 2) * endpoint
        return np.linspace(min_level, max_level, num=num, endpoint=endpoint)

    @cached_property
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> quantizer.thresholds
            array([-2.,  0.,  2.])
        """
        return (self.levels + self.quantization_step / 2)[:-1]

    def mean_squared_error(
        self,
        input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        input_range: tuple[float, float],
        points_per_interval: int = 4096,
    ) -> float:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer.mean_squared_error(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ... )  # doctest: +FLOAT_CMP
            0.3363025489037716
            >>> uniform_pdf = lambda x: 1/8 * (np.abs(x) <= 4)
            >>> quantizer.mean_squared_error(
            ...     input_pdf=uniform_pdf,
            ...     input_range=(-4, 4),
            ... )  # doctest: +FLOAT_CMP
            0.3333333730891729
            >>> quantizer.quantization_step**2 / 12  # doctest: +FLOAT_CMP
            0.3333333333333333
        """
        return super().mean_squared_error(input_pdf, input_range, points_per_interval)

    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> quantizer.digitize([-2.4,  0.8,  3.2])
            array([0, 2, 3])
        """
        input = np.asarray(input, dtype=float)
        output = np.digitize(input, self.thresholds, right=True)
        return output

    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-4, 4))
            >>> quantizer.quantize([-2.4,  0.8,  3.2])
            array([-3.,  1.,  3.])
        """
        input = np.array(input, dtype=float, ndmin=1)
        delta = self.quantization_step
        if self.choice == "mid-riser":
            quantized = delta * (np.floor(input / delta) + 0.5)
        else:
            quantized = delta * np.floor(input / delta + 0.5)
        output = np.clip(quantized, a_min=self.levels[0], a_max=self.levels[-1])
        return output
