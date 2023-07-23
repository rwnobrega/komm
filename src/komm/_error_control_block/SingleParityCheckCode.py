from functools import cached_property

import numpy as np
from attrs import frozen
from scipy import special

from .BlockCode import BlockCode


@frozen(slots=False)
class SingleParityCheckCode(BlockCode):
    r"""
    Single parity check code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose codewords are obtained by extending $n - 1$ information bits with a single parity-check bit. The repetition code has the following parameters:

    - Length: $n$.
    - Dimension: $k = n - 1$.
    - Redundancy: $m = 1$.
    - Minimum distance: $d = 2$.

    Notes:

        - Its dual is the [repetition code](/ref/RepetitionCode).

    Attributes:

        n (int): The length $n$ of the code. Must be a positive integer.

    Examples:

        >>> code = komm.SingleParityCheckCode(5)
        >>> (code.length, code.dimension, code.redundancy)
        (5, 4, 1)
        >>> code.minimum_distance
        2
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1],
               [0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1],
               [0, 0, 0, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 1, 1, 1]])

        >>> code = komm.SingleParityCheckCode(16)
        >>> code.codeword_weight_distribution
        array([    1,     0,   120,     0,  1820,     0,  8008,     0, 12870,
                   0,  8008,     0,  1820,     0,   120,     0,     1])
        >>> code.coset_leader_weight_distribution
        array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    n: int

    @cached_property
    def check_matrix(self):
        return np.ones((1, self.n), dtype=int)

    @property
    def minimum_distance(self):
        return 2

    @cached_property
    def codeword_weight_distribution(self):
        n = self.n
        codeword_weight_distribution = np.zeros(n + 1, dtype=int)
        for w in range(0, n + 1, 2):
            codeword_weight_distribution[w] = special.comb(n, w, exact=True)
        return codeword_weight_distribution

    @property
    def supported_decoders(self):
        return super().supported_decoders + ["wagner"]
