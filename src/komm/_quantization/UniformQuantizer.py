from collections.abc import Callable
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc


class UniformQuantizer(abc.ScalarQuantizer):
    r"""
    Uniform scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) in which the separation between levels is a constant $\Delta$, called the *quantization step*, and the thresholds are the mid-point between adjacent levels. More precisely, the levels are given by
    $$
        v_i = (i - (L - 1)/2 + \theta) \Delta, \qquad i \in [0 : L),
    $$
    where $\theta \in \mathbb{R}$ is an arbitrary *offset* (normalized by $\Delta$), and the finite thresholds are given by
    $$
        t_i = \frac{v_{i-1} + v_i}{2}, \qquad i \in [1 : L).
    $$
    For more details, see <cite>Say06, Sec. 9.4</cite>.

    Parameters:
        num_levels: The number of quantization levels $L$. It must be greater than $1$.

        step: The quantization step $\Delta$. It must be a positive number.

        offset: The offset $\theta$ of the quantizer. The default value is $0$.

    Examples:
        >>> quantizer = komm.UniformQuantizer(num_levels=4, step=0.5)
        >>> quantizer.levels
        array([-0.75, -0.25,  0.25,  0.75])
        >>> quantizer.thresholds
        array([-0.5,  0. ,  0.5])

        >>> quantizer = komm.UniformQuantizer(num_levels=5, step=0.5)
        >>> quantizer.levels
        array([-1. , -0.5,  0. ,  0.5,  1. ])
        >>> quantizer.thresholds
        array([-0.75, -0.25,  0.25,  0.75])

        >>> quantizer = komm.UniformQuantizer(num_levels=5, step=0.5, offset=2.0)
        >>> quantizer.levels
        array([0. , 0.5, 1. , 1.5, 2. ])
        >>> quantizer.thresholds
        array([0.25, 0.75, 1.25, 1.75])
    """

    def __init__(self, num_levels: int, step: float, offset: float = 0.0):
        if not num_levels > 1:
            raise ValueError("'num_levels' must be greater than 1")
        if not step > 0:
            raise ValueError("'step' must be positive")
        self.num_levels = num_levels
        self.step = step
        self.offset = offset

    @classmethod
    def mid_riser(cls, num_levels: int, step: float):
        r"""
        Constructs a *mid-riser* uniform quantizer. In a mid-riser quantizer, $0$ is always a threshold. It is a symmetric quantizer when $L$ is even.

        Parameters:
            num_levels: The number of quantization levels $L$. It must be greater than $1$.

            step: The quantization step $\Delta$. It must be a positive number.

        Examples:
            >>> quantizer = komm.UniformQuantizer.mid_riser(num_levels=4, step=0.5)
            >>> quantizer.levels
            array([-0.75, -0.25,  0.25,  0.75])
            >>> quantizer.thresholds
            array([-0.5,  0. ,  0.5])

            >>> quantizer = komm.UniformQuantizer.mid_riser(num_levels=5, step=0.5)
            >>> quantizer.levels
            array([-1.25, -0.75, -0.25,  0.25,  0.75])
            >>> quantizer.thresholds
            array([-1. , -0.5,  0. ,  0.5])
        """
        return cls(
            num_levels=num_levels,
            step=step,
            offset=0.0 if num_levels % 2 == 0 else -0.5,
        )

    @classmethod
    def mid_tread(cls, num_levels: int, step: float):
        r"""
        Constructs a *mid-tread* uniform quantizer. In a mid-tread quantizer, $0$ is always a level. It is a symmetric quantizer when $L$ is odd.

        Parameters:
            num_levels: The number of quantization levels $L$. It must be greater than $1$.

            step: The quantization step $\Delta$. It must be a positive number.

        Examples:
            >>> quantizer = komm.UniformQuantizer.mid_tread(num_levels=4, step=0.5)
            >>> quantizer.levels
            array([-1. , -0.5,  0. ,  0.5])
            >>> quantizer.thresholds
            array([-0.75, -0.25,  0.25])

            >>> quantizer = komm.UniformQuantizer.mid_tread(num_levels=5, step=0.5)
            >>> quantizer.levels
            array([-1. , -0.5,  0. ,  0.5,  1. ])
            >>> quantizer.thresholds
            array([-0.75, -0.25,  0.25,  0.75])
        """
        return cls(
            num_levels=num_levels,
            step=step,
            offset=0.0 if num_levels % 2 == 1 else -0.5,
        )

    @cached_property
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=8, step=3.0)
            >>> quantizer.levels
            array([-10.5,  -7.5,  -4.5,  -1.5,   1.5,   4.5,   7.5,  10.5])

            >>> quantizer = komm.UniformQuantizer(num_levels=5, step=2.0)
            >>> quantizer.levels
            array([-4., -2.,  0.,  2.,  4.])
        """
        L, θ, Δ = self.num_levels, self.offset, self.step
        return np.array([(i - (L - 1) / 2 + θ) * Δ for i in range(L)])

    @cached_property
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=8, step=3.0)
            >>> quantizer.thresholds
            array([-9., -6., -3.,  0.,  3.,  6.,  9.])

            >>> quantizer = komm.UniformQuantizer(num_levels=5, step=2.0)
            >>> quantizer.thresholds
            array([-3., -1.,  1.,  3.])
        """
        return (self.levels + self.step / 2)[:-1]

    def mean_squared_error(
        self,
        input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        input_range: tuple[float, float],
        points_per_interval: int = 4096,
    ) -> float:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, step=2.0)
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
            >>> quantizer.step**2 / 12  # doctest: +FLOAT_CMP
            0.3333333333333333
        """
        return super().mean_squared_error(input_pdf, input_range, points_per_interval)

    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, step=2.0)
            >>> quantizer.digitize([-2.4,  0.8,  3.2])
            array([0, 2, 3])
        """
        return super().digitize(input)

    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=4, step=2.0)
            >>> quantizer.quantize([-2.4,  0.8,  3.2])
            array([-3.,  1.,  3.])
        """
        return super().quantize(input)
