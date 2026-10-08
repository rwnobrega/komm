from functools import cache
from math import comb

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .._util.validators import validate_positive_integer
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
class SingleParityCheckCode(BlockCode):
    r"""
    Single parity-check code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose codewords are obtained by extending $n - 1$ information bits with a single parity-check bit. The single parity-check code has the following parameters:

    - Length: $n$
    - Dimension: $k = n - 1$
    - Redundancy: $m = 1$
    - Minimum distance: $d = 2$

    Notes:
        - Its dual is the [repetition code](/ref/RepetitionCode).

    Parameters:
        n: The length $n$ of the code. Must be a positive integer.

    Examples:
        >>> code = komm.SingleParityCheckCode(5)
        >>> (code.length, code.dimension, code.redundancy)
        (5, 4, 1)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1],
               [0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1],
               [0, 0, 0, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 1, 1, 1]])
        >>> code.minimum_distance()
        2

        >>> code = komm.SingleParityCheckCode(16)
        >>> code.codeword_weight_distribution()
        array([    1,     0,   120,     0,  1820,     0,  8008,     0, 12870,
                   0,  8008,     0,  1820,     0,   120,     0,     1])
        >>> code.coset_leader_weight_distribution()
        array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """

    def __init__(self, n: int) -> None:
        self.n = validate_positive_integer(n, "n")
        super().__init__(check_matrix=np.ones((1, self.n), dtype=int))

    def __repr__(self) -> str:
        args = f"n={self.n}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 2

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        n = self.n
        codeword_weight_distribution = np.zeros(n + 1, dtype=int)
        for w in range(0, n + 1, 2):
            codeword_weight_distribution[w] = comb(n, w)
        return codeword_weight_distribution
