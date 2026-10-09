from functools import cache

from .._util.docs import mkdocstrings
from .._util.validators import validate_bool, validate_integer
from .HammingCode import hamming_parity_submatrix
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
class SimplexCode(SystematicBlockCode):
    r"""
    Simplex (maximum-length) code. For a given parameter $\kappa \geq 2$, it is the [linear block code](/ref/BlockCode) with generator matrix whose columns are all the $2^\kappa - 1$ nonzero binary $\kappa$-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: $n = 2^\kappa - 1$
    - Dimension: $k = \kappa$
    - Redundancy: $m = 2^\kappa - \kappa - 1$
    - Minimum distance: $d = 2^{\kappa - 1}$

    In its lengthened version, the simplex code has the following parameters:

    - Length: $n = 2^\kappa$
    - Dimension: $k = \kappa + 1$
    - Redundancy: $m = 2^\kappa - \kappa - 1$
    - Minimum distance: $d = 2^{\kappa - 1}$

    For more details, see [EC Zoo: Simplex code](https://errorcorrectionzoo.org/c/simplex).

    Notes:
        - For $\kappa = 2$ it reduces to the [single parity-check code](/ref/SingleParityCheckCode) of length $3$; in its lengthened version, of length $4$.
        - Its lengthened version is equivalent to the [Reed–Muller code](/ref/ReedMullerCode) with parameters $(1, \kappa)$.
        - Its dual is the [Hamming code](/ref/HammingCode); the dual of its lengthened version is the extended Hamming code.
        - Simplex codes are constant-weight codes; their lengthened versions are not.

    Parameters:
        kappa: The parameter $\kappa$ of the code. Must satisfy $\kappa \geq 2$.
        lengthened: Whether to use the lengthened version of the simplex code. Default is `False`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode), with the information set on the left.

    Examples:
        >>> code = komm.SimplexCode(3)
        >>> (code.length, code.dimension, code.redundancy)
        (7, 3, 4)
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1, 0, 1],
               [0, 1, 0, 1, 0, 1, 1],
               [0, 0, 1, 0, 1, 1, 1]])
        >>> code.check_matrix
        array([[1, 1, 0, 1, 0, 0, 0],
               [1, 0, 1, 0, 1, 0, 0],
               [0, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 0, 0, 0, 1]])
        >>> code.minimum_distance()
        4

        >>> code = komm.SimplexCode(3, lengthened=True)
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

    def __init__(self, kappa: int, *, lengthened: bool = False) -> None:
        self.kappa = validate_integer(kappa, low=2)
        self.lengthened = validate_bool(lengthened)
        P = hamming_parity_submatrix(self.kappa, self.lengthened).T
        super().__init__(parity_submatrix=P)

    def __repr__(self) -> str:
        args = f"kappa={self.kappa}, lengthened={self.lengthened}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 2 ** (self.kappa - 1)
