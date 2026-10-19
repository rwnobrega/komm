from functools import cache
from math import comb
from typing import Literal

import numpy as np

from .._util.docs import mkdocstrings
from .._util.validators import validate_choice, validate_positive_integer
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
class SingleParityCheckCode(SystematicBlockCode):
    r"""
    Single parity-check code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose codewords are obtained by extending $n - 1$ information bits with a single parity-check bit. The single parity-check code has the following parameters:

    - Length: $n$
    - Dimension: $k = n - 1$
    - Redundancy: $m = 1$
    - Minimum distance: $d = 2$

    For more details, see <cite>LC04, Sec. 3.7</cite> and [EC Zoo: Single parity-check code](https://errorcorrectionzoo.org/c/parity_check).

    Notes:
        - Its dual is the [repetition code](/ref/RepetitionCode).

    Parameters:
        n: The length $n$ of the code. Must be a positive integer.
        information_set: The side of the information set. Either `'left'` or `'right'`. Default is `'left'`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode).

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
        [1, 0, 120, 0, 1820, 0, 8008, 0, 12870, 0, 8008, 0, 1820, 0, 120, 0, 1]
        >>> code.coset_leader_weight_distribution()
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    def __init__(
        self,
        n: int,
        *,
        information_set: Literal["left", "right"] = "left",
    ) -> None:
        self.n = validate_positive_integer(n)
        side = validate_choice(information_set, ("left", "right"))
        P = np.ones((self.n - 1, 1), dtype=int)
        super().__init__(parity_submatrix=P, information_set=side)

    def __repr__(self) -> str:
        left = np.array_equal(self.information_set, np.arange(self.dimension))
        side = "left" if left else "right"
        args = f"n={self.n}, information_set={side!r}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 2

    @cache
    def codeword_weight_distribution(self) -> list[int]:
        n = self.n
        codeword_weight_distribution = [0] * (n + 1)
        for w in range(0, n + 1, 2):
            codeword_weight_distribution[w] = comb(n, w)
        return codeword_weight_distribution
