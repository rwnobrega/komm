from functools import cache
from math import comb
from typing import Literal

import numpy as np

from .._util.docs import mkdocstrings
from .._util.validators import validate_choice, validate_positive_integer
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
class RepetitionCode(SystematicBlockCode):
    r"""
    Repetition code. For a given length $n \geq 1$, it is the [linear block code](/ref/BlockCode) whose only two codewords are $00 \cdots 0$ and $11 \cdots 1$. The repetition code has the following parameters:

    - Length: $n$
    - Dimension: $k = 1$
    - Redundancy: $m = n - 1$
    - Minimum distance: $d = n$

    For more details, see <cite>LC04, Sec. 3.7</cite> and [EC Zoo: Repetition code](https://errorcorrectionzoo.org/c/repetition).

    Notes:
        - Its dual is the [single parity-check code](/ref/SingleParityCheckCode).

    Parameters:
        n: The length $n$ of the code. Must be a positive integer.
        information_set: The side of the information set. Either `'left'` or `'right'`. Default is `'left'`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode).

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
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        >>> code.coset_leader_weight_distribution()
        [1, 16, 120, 560, 1820, 4368, 8008, 11440, 6435, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    def __init__(
        self,
        n: int,
        *,
        information_set: Literal["left", "right"] = "left",
    ) -> None:
        self.n = validate_positive_integer(n)
        side = validate_choice(information_set, ("left", "right"))
        P = np.ones((1, self.n - 1), dtype=int)
        super().__init__(parity_submatrix=P, information_set=side)

    def __repr__(self) -> str:
        left = np.array_equal(self.information_set, np.arange(self.dimension))
        side = "left" if left else "right"
        args = f"n={self.n}, information_set={side!r}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return self.n

    @cache
    def coset_leader_weight_distribution(self) -> list[int]:
        n = self.n
        coset_leader_weight_distribution = [0] * (n + 1)
        for w in range((n + 1) // 2):
            coset_leader_weight_distribution[w] = comb(n, w)
        if n % 2 == 0:
            coset_leader_weight_distribution[n // 2] = comb(n, n // 2) // 2
        return coset_leader_weight_distribution
