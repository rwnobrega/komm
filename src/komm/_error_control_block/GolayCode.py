from functools import cached_property

from attrs import frozen

from .lib import golay_parity_submatrix
from .SystematicBlockCode import SystematicBlockCode


@frozen
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

    Attributes:

        extended: If `True`, constructs the code in extended version. The default value is `False`.

    This function returns the code in systematic form, with the information set on the left.

    Examples:

        >>> code = komm.GolayCode()
        >>> (code.length, code.dimension, code.redundancy)
        (23, 12, 11)
        >>> code.minimum_distance
        7

        >>> code = komm.GolayCode(extended=True)
        >>> (code.length, code.dimension, code.redundancy)
        (24, 12, 12)
        >>> code.minimum_distance
        8
    """
    extended: bool = False

    @cached_property
    def parity_submatrix(self):
        return golay_parity_submatrix(self.extended)

    @property
    def minimum_distance(self):
        return 8 if self.extended else 7
