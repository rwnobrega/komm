import itertools

import numpy as np

from .BlockCode import BlockCode


class HammingCode(BlockCode):
    r"""
    Hamming code. For a given redundancy $m$, it is the [linear block code](/ref/BlockCode) with parity-check matrix whose columns are all the $2^m - 1$ nonzero binary $m$-tuples. The Hamming code has the following parameters:

    - Length: $n = 2^m - 1$
    - Redundancy: $m$
    - Dimension: $k = 2^m - m - 1$
    - Minimum distance: $d = 3$

    This class constructs the code in systematic form, with the information set on the left.

    For more details, see <cite>LC04, Sec. 4.1</cite>.

    Notes:

        - For $m = 2$ it reduces to the [repetition code](/ref/RepetitionCode) of length $3$.
        - Its dual is the [simplex code](/ref/SimplexCode).
        - Hamming codes are perfect codes.

    Examples:

        >>> code = komm.HammingCode(3)
        >>> (code.length, code.dimension, code.minimum_distance)
        (7, 4, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1, 0],
               [0, 1, 0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1]])
        >>> code.parity_check_matrix
        array([[1, 1, 0, 1, 1, 0, 0],
               [1, 0, 1, 1, 0, 1, 0],
               [0, 1, 1, 1, 0, 0, 1]])
        >>> code.encode([1, 0, 1, 1])
        array([1, 0, 1, 1, 0, 1, 0])
        >>> code.decode([0, 1, 0, 0, 0, 1, 1])
        array([1, 1, 0, 0])

        >>> code = komm.HammingCode(3, extended=True)
        >>> (code.length, code.dimension, code.minimum_distance)
        (8, 4, 4)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1, 0, 1],
               [0, 1, 0, 0, 1, 0, 1, 1],
               [0, 0, 1, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 1, 0]])
        >>> code.parity_check_matrix
        array([[1, 1, 0, 1, 1, 0, 0, 0],
               [1, 0, 1, 1, 0, 1, 0, 0],
               [0, 1, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 0, 0, 0, 0, 1]])
        >>> code.encode([1, 0, 1, 1])
        array([1, 0, 1, 1, 0, 1, 0, 0])
        >>> code.decode([0, 1, 0, 0, 0, 1, 1, 0])
        array([1, 1, 0, 0])
    """

    def __init__(self, m, extended=False):
        r"""
        Constructor for the class.

        Parameters:

            m (int): The redundancy $m$ of the code. Must satisfy $m \geq 2$.

            extended (Optional[bool]): If `True`, constructs the code in extended version. The default value is `False`.
        """
        P = self._hamming_parity_submatrix(m)
        if extended:
            P = BlockCode._extended_parity_submatrix(P)
        super().__init__(parity_submatrix=P)
        self._minimum_distance = 4 if extended else 3
        self._m = m
        self._extended = extended

    def __repr__(self):
        args = "{}".format(self._m)
        if self._extended:
            args += ", extended=True"
        return "{}({})".format(self.__class__.__name__, args)

    @staticmethod
    def _hamming_parity_submatrix(m):
        parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
        i = 0
        for w in range(2, m + 1):
            for idx in itertools.combinations(range(m), w):
                parity_submatrix[i, list(idx)] = 1
                i += 1
        return parity_submatrix
