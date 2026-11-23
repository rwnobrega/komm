from functools import cache

import numpy as np

from .._util.docs import mkdocstrings
from .._util.validators import validate_bool
from ..types import Array2D
from .SystematicBlockCode import SystematicBlockCode
from .util import extended_parity_submatrix


@mkdocstrings(filters=["!.*"])
class GolayCode(SystematicBlockCode):
    r"""
    Golay code. Its extended version is the [linear block code](/ref/BlockCode) with parity submatrix $P = J - A$, where $J$ is the $12 \times 12$ all-ones matrix and $A$ is the adjacency matrix of the icosahedron, with vertices labeled so that vertex $i \in [0 : 10)$ is adjacent to vertices $i \pm 1$ and $i \pm 2$ (modulo $10$), and vertices $10$ and $11$ are adjacent to the even and odd ones, respectively. That is,
    $$
    P = \begin{bmatrix}
        1 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 \\\\
        0 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 1 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 1 \\\\
        1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\\\
        1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 1 \\\\
        1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 \\\\
        1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 \\\\
        1 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 \\\\
        0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 0 \\\\
        0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 \\\\
        1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 1
    \end{bmatrix}.
    $$

    The Golay code is obtained by deleting the last position of its extended version.

    The Golay code has the following parameters:

    - Length: $n = 23$
    - Dimension: $k = 12$
    - Redundancy: $m = 11$
    - Minimum distance: $d = 7$

    In its extended version, the Golay code has the following parameters:

    - Length: $n = 24$
    - Dimension: $k = 12$
    - Redundancy: $m = 12$
    - Minimum distance: $d = 8$

    For more details, see <cite>LC04, Sec. 4.6</cite>, [EC Zoo: Golay code](https://errorcorrectionzoo.org/c/golay), and [Wikipedia: Binary Golay code](https://en.wikipedia.org/wiki/Binary_Golay_code), which uses the same matrix.

    Notes:
        - The Golay code is a perfect code; its extended version is not.
        - Its extended version is self-dual.

    Parameters:
        extended: Whether to use the extended version of the Golay code. The default value is `False`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode), with the information set on the left.

    Examples:
        >>> code = komm.GolayCode()
        >>> (code.length, code.dimension, code.redundancy)
        (23, 12, 11)
        >>> code.minimum_distance()
        7
        >>> (code.packing_radius(), code.covering_radius())
        (3, 3)

        >>> code = komm.GolayCode(extended=True)
        >>> (code.length, code.dimension, code.redundancy)
        (24, 12, 12)
        >>> code.minimum_distance()
        8
        >>> (code.packing_radius(), code.covering_radius())
        (3, 4)
    """

    def __init__(self, *, extended: bool = False) -> None:
        self.extended = validate_bool(extended)
        super().__init__(parity_submatrix=golay_parity_submatrix(self.extended))

    def __repr__(self) -> str:
        args = f"extended={self.extended}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 8 if self.extended else 7


def golay_parity_submatrix(extended: bool = False) -> Array2D[np.integer]:
    parity_submatrix = np.array([
        [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ])
    return extended_parity_submatrix(parity_submatrix) if extended else parity_submatrix
