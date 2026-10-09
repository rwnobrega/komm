from functools import cache
from itertools import combinations

import numpy as np

from .._util.docs import mkdocstrings
from .._util.validators import validate_bool, validate_integer
from ..types import Array2D
from .SystematicBlockCode import SystematicBlockCode
from .util import extended_parity_submatrix


@mkdocstrings(filters=["!.*"])
class HammingCode(SystematicBlockCode):
    r"""
    Hamming code. For a given parameter $\mu \geq 2$, it is the [linear block code](/ref/BlockCode) with check matrix whose columns are all the $2^\mu - 1$ nonzero binary $\mu$-tuples. The Hamming code has the following parameters:

    - Length: $n = 2^\mu - 1$
    - Dimension: $k = 2^\mu - \mu - 1$
    - Redundancy: $m = \mu$
    - Minimum distance: $d = 3$

    In its extended version, the Hamming code has the following parameters:

    - Length: $n = 2^\mu$
    - Dimension: $k = 2^\mu - \mu - 1$
    - Redundancy: $m = \mu + 1$
    - Minimum distance: $d = 4$

    For more details, see <cite>LC04, Sec. 4.1</cite> and [EC Zoo: Hamming code](https://errorcorrectionzoo.org/c/hamming).

    Notes:
        - For $\mu = 2$ it reduces to the [repetition code](/ref/RepetitionCode) of length $3$; in its extended version, of length $4$.
        - Its extended version is equivalent to the [Reed–Muller code](/ref/ReedMullerCode) with parameters $(\mu - 2, \mu)$.
        - Its dual is the [simplex code](/ref/SimplexCode); the dual of its extended version is the lengthened simplex code.
        - Hamming codes are perfect codes; their extended versions are not.

    Parameters:
        mu: The parameter $\mu$ of the code. Must satisfy $\mu \geq 2$.
        extended: Whether to use the extended version of the Hamming code. Default is `False`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode), with the information set on the left.

    Examples:
        >>> code = komm.HammingCode(3)
        >>> (code.length, code.dimension, code.redundancy)
        (7, 4, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1, 0],
               [0, 1, 0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 0, 1, 1, 0, 0],
               [1, 0, 1, 1, 0, 1, 0],
               [0, 1, 1, 1, 0, 0, 1]])
        >>> code.minimum_distance()
        3

        >>> code = komm.HammingCode(3, extended=True)
        >>> (code.length, code.dimension, code.redundancy)
        (8, 4, 4)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1, 0, 1],
               [0, 1, 0, 0, 1, 0, 1, 1],
               [0, 0, 1, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[1, 1, 0, 1, 1, 0, 0, 0],
               [1, 0, 1, 1, 0, 1, 0, 0],
               [0, 1, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 0, 0, 0, 0, 1]])
        >>> code.minimum_distance()
        4
    """

    def __init__(self, mu: int, *, extended: bool = False) -> None:
        self.mu = validate_integer(mu, low=2)
        self.extended = validate_bool(extended)
        P = hamming_parity_submatrix(self.mu, self.extended)
        super().__init__(parity_submatrix=P)

    def __repr__(self) -> str:
        args = f"mu={self.mu}, extended={self.extended}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 4 if self.extended else 3


def hamming_parity_submatrix(m: int, extended: bool = False) -> Array2D[np.integer]:
    parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
    i = 0
    for w in range(2, m + 1):
        for idx in combinations(range(m), w):
            parity_submatrix[i, list(idx)] = 1
            i += 1
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix
