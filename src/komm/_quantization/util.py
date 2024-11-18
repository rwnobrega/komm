from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

from .AbstractScalarQuantizer import AbstractScalarQuantizer


def mean_squared_quantization_error(
    quantizer: AbstractScalarQuantizer,
    input_pdf: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    input_range: Tuple[float, float],
    points_per_interval: int,
) -> np.float64:
    # See [Say06, eq. (9.3)].
    x_min, x_max = input_range
    thresholds = np.concatenate(([x_min], quantizer.thresholds, [x_max]))
    mse = np.float64(0.0)
    for i, level in enumerate(quantizer.levels):
        left, right = thresholds[i], thresholds[i + 1]
        x = np.linspace(left, right, num=points_per_interval, dtype=np.float64)
        pdf = input_pdf(x)
        integrand: npt.NDArray[np.float64] = (x - level) ** 2 * pdf
        integral = np.float64(np.trapezoid(integrand, x))
        mse += integral
    return mse


def lloyd_max_quantizer(
    input_pdf: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    num_levels: int,
    input_range: Tuple[float, float],
    points_per_interval: int,
    max_iter: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
            x = np.linspace(left, right, num=points_per_interval, dtype=np.float64)
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
