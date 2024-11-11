import math
from functools import cached_property

import numpy as np
from attrs import frozen

from .BlockCode import BlockCode


@frozen
class RepetitionCode(BlockCode):
    r"""
    Repetition code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose only two codewords are $00 \cdots 0$ and $11 \cdots 1$. The repetition code has the following parameters:

    - Length: $n$
    - Dimension: $k = 1$
    - Redundancy: $m = n - 1$
    - Minimum distance: $d = n$

    Notes:
        - Its dual is the [single parity check code](/ref/SingleParityCheckCode).

    Attributes:
        n: The length $n$ of the code. Must be a positive integer.

    Examples:
        >>> code = komm.RepetitionCode(5)
        >>> (code.length, code.dimension, code.redundancy)
        (5, 1, 4)
        >>> code.minimum_distance
        5
        >>> code.generator_matrix
        array([[1, 1, 1, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 0, 0, 0],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 1, 0],
               [1, 0, 0, 0, 1]])

        >>> code = komm.RepetitionCode(16)
        >>> code.codeword_weight_distribution
        array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> code.coset_leader_weight_distribution
        array([    1,    16,   120,   560,  1820,  4368,  8008, 11440,  6435,
                   0,     0,     0,     0,     0,     0,     0,     0])
    """

    n: int

    @cached_property
    def generator_matrix(self):
        return np.ones((1, self.n), dtype=int)

    @property
    def minimum_distance(self):
        return self.n

    @cached_property
    def coset_leader_weight_distribution(self):
        n = self.n
        coset_leader_weight_distribution = np.zeros(n + 1, dtype=int)
        for w in range((n + 1) // 2):
            coset_leader_weight_distribution[w] = math.comb(n, w)
        if n % 2 == 0:
            coset_leader_weight_distribution[n // 2] = math.comb(n, n // 2) // 2
        return coset_leader_weight_distribution

    @property
    def default_decoder(self):
        return "majority_logic_repetition_code"

    @classmethod
    def supported_decoders(cls):
        return cls.__base__.supported_decoders() + ["majority_logic_repetition_code"]  # type: ignore
