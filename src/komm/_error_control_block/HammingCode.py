from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .SystematicBlockCode import SystematicBlockCode
from .util import extended_parity_submatrix


@mkdocstrings(filters=["!.*"])
@dataclass(eq=False)
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

    For more details, see <cite>LC04, Sec. 4.1</cite>.

    Notes:
        - For $\mu = 2$ it reduces to the [repetition code](/ref/RepetitionCode) of length $3$.
        - Its dual is the [simplex code](/ref/SimplexCode).
        - Hamming codes are perfect codes.

    Attributes:
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

    mu: int
    extended: bool = False

    def __post_init__(self):
        if not self.mu >= 2:
            raise ValueError("'mu' must be at least 2")
        super().__init__(
            parity_submatrix=hamming_parity_submatrix(self.mu, self.extended)
        )

    @cache
    def minimum_distance(self) -> int:
        return 4 if self.extended else 3


def hamming_parity_submatrix(m: int, extended: bool = False) -> npt.NDArray[np.integer]:
    parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
    i = 0
    for w in range(2, m + 1):
        for idx in combinations(range(m), w):
            parity_submatrix[i, list(idx)] = 1
            i += 1
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix
