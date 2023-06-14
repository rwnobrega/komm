from .BlockCode import BlockCode
from .HammingCode import HammingCode


class SimplexCode(BlockCode):
    r"""
    Simplex (maximum-length) code. For a given dimension $k$, it is the [linear block code](/ref/BlockCode) with generator matrix whose columns are all the $2^k - 1$ nonzero binary $k$-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: $n = 2^k - 1$
    - Dimension: $k$
    - Redundancy: $m = 2^k - k - 1$
    - Minimum distance: $d = 2^{k - 1}$

    This class constructs the code in systematic form, with the information set on the left.

    Notes:

        - For $k = 2$ it reduces to the [single parity check code](/ref/SingleParityCheckCode) of length $3$.
        - Its dual is the [Hamming code](/ref/HammingCode).
        - Simplex codes are constant-weight codes.
    """

    def __init__(self, k):
        r"""
        Constructor for the class.

        Parameters:

            k (int): The dimension $k$ of the code. Must satisfy $k \geq 2$.

        Examples:

            >>> code = komm.SimplexCode(3)
            >>> (code.length, code.dimension, code.minimum_distance)
            (7, 3, 4)
            >>> code.generator_matrix
            array([[1, 0, 0, 1, 1, 0, 1],
                   [0, 1, 0, 1, 0, 1, 1],
                   [0, 0, 1, 0, 1, 1, 1]])
            >>> code.parity_check_matrix
            array([[1, 1, 0, 1, 0, 0, 0],
                   [1, 0, 1, 0, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 0],
                   [1, 1, 1, 0, 0, 0, 1]])
            >>> code.encode([1, 0, 1])
            array([1, 0, 1, 1, 0, 1, 0])
            >>> code.decode([1, 0, 1, 1, 1, 1, 0])
            array([1, 0, 1])
        """
        P = HammingCode._hamming_parity_submatrix(k).T
        super().__init__(parity_submatrix=P)
        self._minimum_distance = 2 ** (k - 1)
        self._k = k

    def __repr__(self):
        args = "{}".format(self._k)
        return "{}({})".format(self.__class__.__name__, args)
