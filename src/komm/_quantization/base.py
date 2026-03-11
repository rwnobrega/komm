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
        The quantizer levels $y_0, y_1, \ldots, y_{L-1}$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer finite thresholds $\lambda_1, \lambda_2, \ldots, \lambda_{L-1}$.
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
        input = np.asarray(input, dtype=float)
        output = np.digitize(input, self.thresholds, right=False)
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
        Computes the mean squared (quantization) error (MSE) of the quantizer for a given input pdf. It is defined as
        $$
            \mse = \int_{-\infty}^{\infty} (y - x)^2 f_X(x) \, dx
        $$
        where $y$ is the quantized signal and $f_X(x)$ is the pdf of the input signal.

        Parameters:
            input_pdf: The pdf $f_X(x)$ of the input signal.
            input_range: The range $(x_\mathrm{min}, x_\mathrm{max})$ of the input signal.
            points_per_interval: The number of points per interval for numerical integration (default: `4096`).

        Returns:
            mse: The mean square quantization error.
        """
        # See [Say06, eq. (9.3)].
        x_min, x_max = input_range
        λ = np.concatenate(([x_min], self.thresholds, [x_max]))
        mse = 0.0
        for i, level in enumerate(self.levels):
            x = np.linspace(λ[i], λ[i + 1], num=points_per_interval, dtype=float)
            integrand = (level - x) ** 2 * input_pdf(x)
            mse += np.trapezoid(integrand, x)
        return float(mse)
