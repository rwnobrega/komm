from functools import cache
from typing import Literal

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .AbstractScalarQuantizer import AbstractScalarQuantizer


@frozen
class UniformQuantizer(AbstractScalarQuantizer):
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

    _num_levels: int = field(alias="num_levels")
    input_range: tuple[float, float] = field(default=(-1.0, 1.0))
    choice: Literal["mid-riser", "mid-tread"] = field(default="mid-riser")

    def __attrs_post_init__(self) -> None:
        if self.num_levels < 2:
            raise ValueError("'num_levels' must be greater than 1")
        if self.choice not in ["mid-riser", "mid-tread"]:
            raise ValueError("'choice' must be either 'mid-riser' or 'mid-tread'")

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    @cache
    def quantization_step(self) -> float:
        r"""
        The quantization step $\Delta$.
        """
        x_min, x_max = self.input_range
        delta = (x_max - x_min) / self.num_levels
        return delta

    @property
    @cache
    def levels(self) -> npt.NDArray[np.float64]:
        r"""
        The quantizer levels $v_0, v_1, \ldots, v_{L-1}$.
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
    def thresholds(self) -> npt.NDArray[np.float64]:
        r"""
        The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$.
        """
        return (self.levels + self.quantization_step / 2)[:-1]

    def __call__(self, input_signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        input_signal = np.array(input_signal, dtype=float, ndmin=1)
        delta = self.quantization_step
        if self.choice == "mid-riser":
            quantized = delta * (np.floor(input_signal / delta) + 0.5)
        else:
            quantized = delta * np.floor(input_signal / delta + 0.5)
        output_signal = np.clip(quantized, a_min=self.levels[0], a_max=self.levels[-1])
        return output_signal
