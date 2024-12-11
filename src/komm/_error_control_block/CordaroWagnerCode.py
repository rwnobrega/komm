from functools import cache, cached_property

import numpy as np
import numpy.typing as npt
from attrs import frozen

from .BlockCode import BlockCode
from .matrices import cordaro_wagner_generator_matrix


@frozen
class CordaroWagnerCode(BlockCode):
    r"""
    Cordaro–Wagner code. For a given length $n \geq 2$, it is the [linear block code](/ref/BlockCode) with dimension $k = 2$ which is optimum for the [BSC](/ref/BinarySymmetricChannel) with sufficiently small crossover probability. For more details, see <cite>CW67</cite>.

    - Length: $n$
    - Dimension: $k = 2$
    - Redundancy: $m = n - 2$
    - Minimum distance: $d = \left\lceil 2n / 3 \right\rceil - 1$

    Attributes:
        n: The length $n$ of the code. Must satisfy $n \geq 2$.

    Examples:
        >>> code = komm.CordaroWagnerCode(11)
        >>> (code.length, code.dimension, code.redundancy)
        (11, 2, 9)
        >>> code.generator_matrix
        array([[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        >>> code.minimum_distance()
        7
        >>> code.codeword_weight_distribution()
        array([1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0])
        >>> code.coset_leader_weight_distribution()
        array([  1,  11,  55, 165, 226,  54,   0,   0,   0,   0,   0,   0])
    """

    n: int

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        return cordaro_wagner_generator_matrix(self.n)

    @cache
    def minimum_distance(self) -> int:
        return int(np.ceil(2 * self.n / 3)) - 1
