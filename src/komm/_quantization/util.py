from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from . import base


def mean_squared_quantization_error(
    quantizer: base.ScalarQuantizer,
    input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    input_range: tuple[float, float],
    points_per_interval: int,
) -> float:
    # See [Say06, eq. (9.3)].
    x_min, x_max = input_range
    thresholds = np.concatenate(([x_min], quantizer.thresholds, [x_max]))
    mse = 0.0
    for i, level in enumerate(quantizer.levels):
        left, right = thresholds[i], thresholds[i + 1]
        x = np.linspace(left, right, num=points_per_interval, dtype=float)
        pdf = input_pdf(x)
        integrand: npt.NDArray[np.floating] = (x - level) ** 2 * pdf
        integral = np.trapezoid(integrand, x)
        mse += float(integral)
    return mse
