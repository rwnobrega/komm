from functools import cache, cached_property

from attrs import frozen

from .matrices import hamming_parity_submatrix
from .SystematicBlockCode import SystematicBlockCode


@frozen
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

    This function returns the code in systematic form, with the information set on the left.

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

    @cached_property
    def parity_submatrix(self):
        return hamming_parity_submatrix(self.mu, self.extended)

    @cache
    def minimum_distance(self):
        return 4 if self.extended else 3
