import numpy as np

from .BlockCode import BlockCode


class GolayCode(BlockCode):
    r"""
    Binary Golay code. It has the following parameters:

    - Length: $23$
    - Dimension: $12$
    - Minimum distance: $7$

    This class constructs the code in systematic form, with the information set on the left.

    .. rubric:: Decoding methods

    [[decoding_methods]]

    Notes:

        - The binary Golay code is a perfect code.

    Examples:

        >>> code = komm.GolayCode()
        >>> (code.length, code.dimension, code.minimum_distance)
        (23, 12, 7)
        >>> recvword = np.zeros(23, dtype=int); recvword[[2, 10, 19]] = 1
        >>> code.decode(recvword)  # Golay code can correct up to 3 errors.
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> recvword = np.zeros(23, dtype=int); recvword[[2, 3, 10, 19]] = 1
        >>> code.decode(recvword)  # Golay code cannot correct more than 3 errors.
        array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])

        >>> code = komm.GolayCode(extended=True)
        >>> (code.length, code.dimension, code.minimum_distance)
        (24, 12, 8)
    """

    def __init__(self, extended=False):
        r"""
        Constructor for the class.

        Parameters:

            extended (Optional[bool]): If `True`, constructs the code in extended version. The default value is `False`.
        """
        P = GolayCode._golay_parity_submatrix()
        if extended:
            P = BlockCode._extended_parity_submatrix(P)
        super().__init__(parity_submatrix=P)
        self._minimum_distance = 8 if extended else 7
        self._extended = extended

    def __repr__(self):
        args = ""
        if self._extended:
            args += ", extended=True"
        return "{}({})".format(self.__class__.__name__, args)

    @staticmethod
    def _golay_parity_submatrix():
        return np.array(
            [
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
            ]
        )
