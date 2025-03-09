from dataclasses import dataclass
from functools import cache
from math import comb

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
@dataclass(eq=False)
class RepetitionCode(BlockCode):
    r"""
    Repetition code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose only two codewords are $00 \cdots 0$ and $11 \cdots 1$. The repetition code has the following parameters:

    - Length: $n$
    - Dimension: $k = 1$
    - Redundancy: $m = n - 1$
    - Minimum distance: $d = n$

    Notes:
        - Its dual is the [single parity-check code](/ref/SingleParityCheckCode).

    Attributes:
        n: The length $n$ of the code. Must be a positive integer.

    Examples:
        >>> code = komm.RepetitionCode(5)
        >>> (code.length, code.dimension, code.redundancy)
        (5, 1, 4)
        >>> code.generator_matrix
        array([[1, 1, 1, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 0, 0, 0],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 1, 0],
               [1, 0, 0, 0, 1]])
        >>> code.minimum_distance()
        5

        >>> code = komm.RepetitionCode(16)
        >>> code.codeword_weight_distribution()
        array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> code.coset_leader_weight_distribution()
        array([    1,    16,   120,   560,  1820,  4368,  8008, 11440,  6435,
                   0,     0,     0,     0,     0,     0,     0,     0])
    """

    n: int

    def __post_init__(self):
        if not self.n >= 1:
            raise ValueError("n must be a positive integer")
        super().__init__(generator_matrix=np.ones((1, self.n), dtype=int))

    @cache
    def minimum_distance(self) -> int:
        return self.n

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        n = self.n
        coset_leader_weight_distribution = np.zeros(n + 1, dtype=int)
        for w in range((n + 1) // 2):
            coset_leader_weight_distribution[w] = comb(n, w)
        if n % 2 == 0:
            coset_leader_weight_distribution[n // 2] = comb(n, n // 2) // 2
        return coset_leader_weight_distribution
