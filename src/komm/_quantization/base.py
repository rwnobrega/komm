from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property

import numpy as np
import numpy.typing as npt


class ScalarQuantizer(ABC):
    @cached_property
    @abstractmethod
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer levels $v_0, v_1, \ldots, v_{L-1}$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$.
        """
        raise NotImplementedError

    @abstractmethod
    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Returns the quantization indices for the input signal.

        Parameters:
            input: The input signal $x$ to be digitized.

        Returns:
            output: The integer indices of the quantization levels for each input sample.
        """
        tiled = np.tile(input, reps=(self.thresholds.size, 1)).transpose()
        output = np.sum(tiled >= self.thresholds, axis=1)
        return output

    @abstractmethod
    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Quantizes the input signal.

        Parameters:
            input: The input signal $x$ to be quantized.

        Returns:
            output: The quantized signal $y$.
        """
        return self.levels[self.digitize(input)]

    @abstractmethod
    def mean_squared_error(
        self,
        input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        input_range: tuple[float, float],
        points_per_interval: int = 4096,
    ) -> float:
        r"""
        Computes the mean squared (quantization) error (MSE) of the quantizer for a given input
        probability density function (pdf). It is defined as
        $$
            \mse = \int_{-\infty}^{\infty} (x - y)^2 f_X(x) \, dx
        $$
        where $y$ is the quantized signal and $f_X(x)$ is the pdf of the input signal.

        Parameters:
            input_pdf: The probability density function $f_X(x)$ of the input signal.
            input_range: The range $(x_\mathrm{min}, x_\mathrm{max})$ of the input signal.
            points_per_interval: The number of points per interval for numerical integration (default: 4096).

        Returns:
            mse: The mean square quantization error.
        """
        # See [Say06, eq. (9.3)].
        x_min, x_max = input_range
        thresholds = np.concatenate(([x_min], self.thresholds, [x_max]))
        mse = 0.0
        for i, level in enumerate(self.levels):
            left, right = thresholds[i], thresholds[i + 1]
            x = np.linspace(left, right, num=points_per_interval, dtype=float)
            pdf = input_pdf(x)
            integrand: npt.NDArray[np.floating] = (x - level) ** 2 * pdf
            integral = np.trapezoid(integrand, x)
            mse += float(integral)
        return mse
