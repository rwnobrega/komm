from functools import cache

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from .._util.docs import mkdocstrings
from .SystematicBlockCode import SystematicBlockCode
from .util import extended_parity_submatrix


@mkdocstrings(filters=["!.*"])
@typechecked
class GolayCode(SystematicBlockCode):
    r"""
    Binary Golay code. It is the [linear block code](/ref/BlockCode) with parity submatrix
    $$
    P = \begin{bmatrix}
        1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\\\
        0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\\\
        1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 1 \\\\
        1 & 1 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 0 & 1 \\\\
        1 & 1 & 1 & 0 & 1 & 1 & 0 & 1 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 1 \\\\
        0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 1 \\\\
        0 & 1 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 1 & 1 \\\\
        1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 1 \\\\
        1 & 0 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & 1 \\\\
        1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1
    \end{bmatrix}
    $$

    The Golay code has the following parameters:

    - Length: $23$
    - Dimension: $12$
    - Minimum distance: $7$

    Notes:
        - The binary Golay code is a perfect code.

    Parameters:
        extended: If `True`, constructs the code in extended version. The default value is `False`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode), with the information set on the left.

    Examples:
        >>> code = komm.GolayCode()
        >>> (code.length, code.dimension, code.redundancy)
        (23, 12, 11)
        >>> code.minimum_distance()
        7

        >>> code = komm.GolayCode(extended=True)
        >>> (code.length, code.dimension, code.redundancy)
        (24, 12, 12)
        >>> code.minimum_distance()
        8
    """

    def __init__(self, extended: bool = False) -> None:
        self.extended = extended
        super().__init__(parity_submatrix=golay_parity_submatrix(extended))

    def __repr__(self) -> str:
        args = f"extended={self.extended}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 8 if self.extended else 7


def golay_parity_submatrix(extended: bool = False) -> npt.NDArray[np.integer]:
    parity_submatrix = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    ])
    return extended_parity_submatrix(parity_submatrix) if extended else parity_submatrix
