from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .ScalarQuantizer import ScalarQuantizer
from .util import lloyd_max_quantizer


def LloydMaxQuantizer(
    input_pdf: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    input_range: tuple[float, float],
    num_levels: int,
) -> ScalarQuantizer:
    r"""
    Lloyd–Max scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) that minimizes the mean-squared error (MSE) between the input signal $X$ and its quantized version. For more details, see <cite>Say06, Sec. 9.6.1</cite>.

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
        >>> quantizer.thresholds.round(3)  # doctest: +SKIP
        array([-1.748, -1.05 , -0.501, -0.   ,  0.501,  1.05 ,  1.748])
    """
    levels, thresholds = lloyd_max_quantizer(
        input_pdf,
        num_levels,
        input_range,
        points_per_interval=1000,
        max_iter=100,
    )
    return ScalarQuantizer(levels=levels, thresholds=thresholds)
