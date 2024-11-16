from functools import cache
from typing import Literal

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .AbstractScalarQuantizer import AbstractScalarQuantizer


@frozen
class UniformQuantizer(AbstractScalarQuantizer):
    r"""
    Uniform scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) in which the separation between levels is constant, $\Delta$, and the thresholds are the mid-point between adjacent levels.

    Attributes:
        num_levels: The number of quantization levels $L$. It must be greater than $1$.

        input_peak: The peak of the input signal $x_\mathrm{p}$. The default value is `1.0`.

        choice: The choice for the uniform quantizer. Must be one of `'unsigned'` | `'mid-riser'` | `'mid-tread'`. The default value is `'mid-riser'`.

    Examples:
        >>> quantizer = komm.UniformQuantizer(num_levels=8)
        >>> quantizer.levels
        array([-0.875, -0.625, -0.375, -0.125,  0.125,  0.375,  0.625,  0.875])
        >>> quantizer.thresholds
        array([-0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75])
        >>> x = np.linspace(-0.5, 0.5, num=11)
        >>> y = quantizer(x)
        >>> np.vstack([x, y])  # doctest: +NORMALIZE_WHITESPACE
        array([[-0.5  , -0.4  , -0.3  , -0.2  , -0.1  ,  0.   ,  0.1  ,  0.2  ,  0.3  ,  0.4  ,  0.5  ],
               [-0.375, -0.375, -0.375, -0.125, -0.125,  0.125,  0.125,  0.125,  0.375,  0.375,  0.625]])

        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='unsigned')
        >>> quantizer.levels
        array([0.  , 0.25, 0.5 , 0.75])
        >>> quantizer.thresholds
        array([0.125, 0.375, 0.625])

        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='mid-riser')
        >>> quantizer.levels
        array([-0.75, -0.25,  0.25,  0.75])
        >>> quantizer.thresholds
        array([-0.5,  0. ,  0.5])

        >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='mid-tread')
        >>> quantizer.levels
        array([-1. , -0.5,  0. ,  0.5])
        >>> quantizer.thresholds
        array([-0.75, -0.25,  0.25])
    """

    num_levels: int  # TODO: Fix type error
    input_peak: float = field(default=1.0)
    choice: Literal["unsigned", "mid-riser", "mid-tread"] = field(default="mid-riser")

    def __attrs_post_init__(self) -> None:
        if self.num_levels < 2:
            raise ValueError("number of levels must be greater than 1")
        if self.choice not in ["unsigned", "mid-riser", "mid-tread"]:
            raise ValueError(
                "parameter 'choice' must be in {'unsigned', 'mid-riser', 'mid-tread'}"
            )

    @property
    @cache
    def quantization_step(self) -> float:
        r"""
        The quantization step $\Delta$.
        """
        d = self.input_peak / self.num_levels
        return d if self.choice == "unsigned" else 2.0 * d

    @property
    @cache
    def levels(self) -> npt.NDArray[np.float64]:
        r"""
        The quantizer levels $v_0, v_1, \ldots, v_{L-1}$.
        """
        num, peak, delta = self.num_levels, self.input_peak, self.quantization_step
        if self.choice == "unsigned":
            min_level, max_level = 0.0, peak
            return np.linspace(min_level, max_level, num=num, endpoint=False)
        elif self.choice == "mid-riser":
            min_level = -peak + (delta / 2) * (num % 2 == 0)
            return np.linspace(min_level, -min_level, num=num, endpoint=(num % 2 == 0))
        else:  # self.choice == "mid-tread"
            min_level = -peak + (delta / 2) * (num % 2 == 1)
            return np.linspace(min_level, -min_level, num=num, endpoint=(num % 2 == 1))

    @property
    @cache
    def thresholds(self) -> npt.NDArray[np.float64]:
        r"""
        The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$.
        """
        return (self.levels + self.quantization_step / 2)[:-1]

    def __call__(self, input_signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        input_signal = np.array(input_signal, dtype=float, ndmin=1)
        delta = self.quantization_step
        if self.choice in ["unsigned", "mid-tread"]:
            quantized = delta * np.floor(input_signal / delta + 0.5)
        else:  # self.choice == "mid-riser"
            quantized = delta * (np.floor(input_signal / delta) + 0.5)
        output_signal = np.clip(quantized, a_min=self.levels[0], a_max=self.levels[-1])
        return output_signal
