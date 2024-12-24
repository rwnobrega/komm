from dataclasses import dataclass, field
from functools import cache
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
        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
        >>> quantizer.levels
        array([-0.75, -0.25,  0.25,  0.75])
        >>> quantizer.thresholds
        array([-0.5,  0. ,  0.5])

        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-tread')
        >>> quantizer.levels
        array([-1. , -0.5,  0. ,  0.5])
        >>> quantizer.thresholds
        array([-0.75, -0.25,  0.25])

        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(0.0, 1.0), choice='mid-tread')
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

    @property
    @cache
    def quantization_step(self) -> float:
        r"""
        The quantization step $\Delta$.

        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
            >>> quantizer.quantization_step
            0.5
        """
        x_min, x_max = self.input_range
        delta = (x_max - x_min) / self.num_levels
        return delta

    @property
    @cache
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
            >>> quantizer.levels
            array([-0.75, -0.25,  0.25,  0.75])
        """
        num = self.num_levels
        x_min, x_max = self.input_range
        delta = (x_max - x_min) / num
        endpoint = (num % 2 == 0) if self.choice == "mid-riser" else (num % 2 == 1)
        min_level = x_min + (delta / 2) * endpoint
        max_level = x_max - (delta / 2) * endpoint
        return np.linspace(min_level, max_level, num=num, endpoint=endpoint)

    @property
    @cache
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
            >>> quantizer.thresholds
            array([-0.5,  0. ,  0.5])
        """
        return (self.levels + self.quantization_step / 2)[:-1]

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
            >>> quantizer([-0.6, 0.2, 0.8])
            array([-0.75,  0.25,  0.75])
        """
        input = np.array(input, dtype=float, ndmin=1)
        delta = self.quantization_step
        if self.choice == "mid-riser":
            quantized = delta * (np.floor(input / delta) + 0.5)
        else:
            quantized = delta * np.floor(input / delta + 0.5)
        output = np.clip(quantized, a_min=self.levels[0], a_max=self.levels[-1])
        return output
