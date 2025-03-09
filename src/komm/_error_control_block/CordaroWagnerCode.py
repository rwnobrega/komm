from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
@dataclass(eq=False)
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

    def __post_init__(self):
        if not self.n >= 2:
            raise ValueError("n must be at least 2")
        super().__init__(generator_matrix=cordaro_wagner_generator_matrix(self.n))

    @cache
    def minimum_distance(self) -> int:
        return int(np.ceil(2 * self.n / 3)) - 1


def cordaro_wagner_generator_matrix(n: int) -> npt.NDArray[np.integer]:
    # See [CW67].
    d = int(np.ceil(2 * n / 3)) - 1
    q = (n + 1) // 3
    return np.hstack((
        np.repeat([[1], [0]], repeats=d - q, axis=1),
        np.repeat([[0], [1]], repeats=q, axis=1),
        np.repeat([[1], [1]], repeats=n - d, axis=1),
    ))
