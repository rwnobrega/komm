from functools import cache
from typing import Literal

import numpy as np

from .._util.docs import mkdocstrings
from .._util.validators import validate_bool, validate_choice, validate_integer
from .HammingCode import hamming_parity_submatrix
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
class SimplexCode(SystematicBlockCode):
    r"""
    Simplex (maximum-length) code. For a given parameter $\mu \geq 2$, it is the [linear block code](/ref/BlockCode) with generator matrix whose columns are all the $2^\mu - 1$ nonzero binary $\mu$-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: $n = 2^\mu - 1$
    - Dimension: $k = \mu$
    - Redundancy: $m = 2^\mu - \mu - 1$
    - Minimum distance: $d = 2^{\mu - 1}$

    In its lengthened version, the simplex code has the following parameters:

    - Length: $n = 2^\mu$
    - Dimension: $k = \mu + 1$
    - Redundancy: $m = 2^\mu - \mu - 1$
    - Minimum distance: $d = 2^{\mu - 1}$

    For more details, see [EC Zoo: Simplex code](https://errorcorrectionzoo.org/c/simplex).

    Notes:
        - For $\mu = 2$ it reduces to the [single parity-check code](/ref/SingleParityCheckCode) of length $3$; in its lengthened version, of length $4$.
        - Its lengthened version is equivalent to the [Reed–Muller code](/ref/ReedMullerCode) with parameters $(\mu, 1)$.
        - Its dual is the [Hamming code](/ref/HammingCode) with the information set on the opposite side; the same holds for its lengthened version and the extended Hamming code.
        - Simplex codes are constant-weight codes; their lengthened versions are not.

    Parameters:
        mu: The parameter $\mu$ of the code. Must satisfy $\mu \geq 2$.
        lengthened: Whether to use the lengthened version of the simplex code. The default value is `False`.
        information_set: The side of the information set. Either `'left'` or `'right'`. The default value is `'left'`.

    This class represents the code in [systematic form](/ref/SystematicBlockCode).

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

    def __init__(
        self,
        mu: int,
        *,
        lengthened: bool = False,
        information_set: Literal["left", "right"] = "left",
    ) -> None:
        self.mu = validate_integer(mu, low=2)
        self.lengthened = validate_bool(lengthened)
        side = validate_choice(information_set, ("left", "right"))
        P = hamming_parity_submatrix(self.mu, self.lengthened).T
        super().__init__(parity_submatrix=P, information_set=side)

    def __repr__(self) -> str:
        left = np.array_equal(self.information_set, np.arange(self.dimension))
        side = "left" if left else "right"
        args = f"mu={self.mu}, lengthened={self.lengthened}, information_set={side!r}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return 2 ** (self.mu - 1)
