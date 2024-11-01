from functools import cached_property

from attrs import frozen

from ._matrices import hamming_parity_submatrix
from .SystematicBlockCode import SystematicBlockCode


@frozen
class SimplexCode(SystematicBlockCode):
    r"""
    Simplex (maximum-length) code. For a given parameter $\kappa \geq 2$, it is the [linear block code](/ref/BlockCode) with generator matrix whose columns are all the $2^\kappa - 1$ nonzero binary $\kappa$-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: $n = 2^\kappa - 1$
    - Dimension: $k = \kappa$
    - Redundancy: $m = 2^\kappa - \kappa - 1$
    - Minimum distance: $d = 2^{\kappa - 1}$

    In its extended version, the simplex code has the following parameters:

    - Length: $n = 2^\kappa$
    - Dimension: $k = \kappa + 1$
    - Redundancy: $m = 2^\kappa - \kappa - 1$
    - Minimum distance: $d = 2^{\kappa - 1}$

    Notes:

        - For $\kappa = 2$ it reduces to the [single parity check code](/ref/SingleParityCheckCode) of length $3$.
        - Its dual is the [Hamming code](/ref/HammingCode).
        - Simplex codes are constant-weight codes.

    Attributes:

        kappa: The parameter $\kappa$ of the code. Must satisfy $\kappa \geq 2$.
        extended: Whether to use the extended version of the Simplex code. Default is `False`.

    This function constructs the code in systematic form, with the information set on the left.

    Examples:

        >>> code = komm.SimplexCode(3)
        >>> (code.length, code.dimension, code.redundancy)
        (7, 3, 4)
        >>> code.minimum_distance
        4
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1, 0, 1],
               [0, 1, 0, 1, 0, 1, 1],
               [0, 0, 1, 0, 1, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 0, 1, 0, 0, 0],
               [1, 0, 1, 0, 1, 0, 0],
               [0, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 0, 0, 0, 1]])

        >>> code = komm.SimplexCode(3, extended=True)
        >>> (code.length, code.dimension, code.redundancy)
        (8, 4, 4)
        >>> code.minimum_distance
        4
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
    """

    kappa: int
    extended: bool = False

    @cached_property
    def parity_submatrix(self):
        return hamming_parity_submatrix(self.kappa, self.extended).T

    @property
    def minimum_distance(self):
        return 2 ** (self.kappa - 1)
