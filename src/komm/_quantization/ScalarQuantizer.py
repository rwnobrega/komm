from collections.abc import Callable
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


class ScalarQuantizer(base.ScalarQuantizer):
    r"""
    General scalar quantizer. It is defined by a list of *levels*, $v_0, v_1, \ldots, v_{L-1}$, and a list of *thresholds*, $t_0, t_1, \ldots, t_L$, satisfying
    $$
        -\infty = t_0 < v_0 < t_1 < v_1 < \cdots < t_{L - 1} < v_{L - 1} < t_L = +\infty.
    $$
    Given an input $x \in \mathbb{R}$, the output of the quantizer is given by $y = v_i$ if and only if $t_i \leq x < t_{i+1}$, where $i \in [0:L)$. For more details, see <cite>Say06, Ch. 9</cite>.

    Attributes:
        levels: The quantizer levels $v_0, v_1, \ldots, v_{L-1}$. It should be a list floats of length $L$.

        thresholds: The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$. It should be a list of floats of length $L - 1$.

    Examples:
        The $5$-level scalar quantizer whose characteristic (input × output) curve is depicted in the figure below has levels
        $$
            v_0 = -2, ~ v_1 = -1, ~ v_2 = 0, ~ v_3 = 1, ~ v_4 = 2,
        $$
        and thresholds
        $$
            t_0 = -\infty, ~ t_1 = -1.5, ~ t_2 = -0.3, ~ t_3 = 0.8, ~ t_4 = 1.4, ~ t_5 = \infty.
        $$

        <figure markdown>
          ![Scalar quantizer example.](/figures/scalar_quantizer_5.svg)
        </figure>

        >>> quantizer = komm.ScalarQuantizer(
        ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
        ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
        ... )
    """

    def __init__(self, levels: npt.ArrayLike, thresholds: npt.ArrayLike) -> None:
        self._levels = np.asarray(levels, dtype=float)
        self._thresholds = np.asarray(thresholds, dtype=float)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.thresholds.size == self.levels.size - 1:
            raise ValueError("'len(thresholds)' must be equal to 'len(levels) - 1'")

        interleaved = np.empty(2 * self.levels.size - 1, dtype=float)
        interleaved[0::2] = self.levels
        interleaved[1::2] = self.thresholds

        if not np.array_equal(np.unique(interleaved), interleaved):
            raise ValueError("invalid values for 'levels' and 'thresholds'")

    def __repr__(self) -> str:
        args = ", ".join([
            f"levels={self.levels.tolist()}",
            f"thresholds={self.thresholds.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.levels
            array([-2., -1.,  0.,  1.,  2.])
        """
        return self._levels

    @cached_property
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.thresholds
            array([-1.5, -0.3,  0.8,  1.4])
        """
        return self._thresholds

    def mean_squared_error(
        self,
        input_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        input_range: tuple[float, float],
        points_per_interval: int = 4096,
    ) -> float:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> gaussian_pdf = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
            >>> quantizer.mean_squared_error(
            ...     input_pdf=gaussian_pdf,
            ...     input_range=(-5, 5),
            ... )  # doctest: +FLOAT_CMP
            0.13598089455499335
        """
        return super().mean_squared_error(input_pdf, input_range, points_per_interval)

    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.digitize([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
            array([0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 4])
        """
        return super().digitize(input)

    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> quantizer = komm.ScalarQuantizer(
            ...     levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
            ...     thresholds=[-1.5, -0.3, 0.8, 1.4],
            ... )
            >>> quantizer.quantize([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
            array([-2., -2., -1., -1., -1.,  0.,  0.,  1.,  2.,  2.,  2.])
        """
        return super().quantize(input)
