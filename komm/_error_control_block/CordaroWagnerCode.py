import numpy as np

from .BlockCode import BlockCode


class CordaroWagnerCode(BlockCode):
    r"""
    Cordaro–Wagner code. It is the $(n, 2)$ linear block code (:obj:`BlockCode`) which is optimum for the BSC with sufficiently small crossover probability.

    References:

        1. :cite:`Cordaro.Wagner.67`

    .. rubric:: Decoding methods

    [[decoding_methods]]

    Examples:

        >>> code = komm.CordaroWagnerCode(11)
        >>> (code.length, code.dimension, code.minimum_distance)
        (11, 2, 7)
        >>> code.generator_matrix
        array([[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        >>> code.codeword_weight_distribution
        array([1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0])
        >>> code.coset_leader_weight_distribution
        array([  1,  11,  55, 165, 226,  54,   0,   0,   0,   0,   0,   0])
    """

    def __init__(self, n):
        r"""
        Constructor for the class.

        Parameters:

            n (:obj:`int`): The length $n$ of the code. Must satisfy $n \geq 2$.
        """
        r = (n + 1) // 3
        s = n - 3 * r
        if s == 1:
            (h, i, j) = (r, r, r + 1)
        elif s == -1:
            (h, i, j) = (r - 1, r, r)
        else:  # s == 0
            (h, i, j) = (r - 1, r, r + 1)
        H = np.repeat([[1], [0]], repeats=h, axis=1)
        I = np.repeat([[0], [1]], repeats=i, axis=1)
        J = np.repeat([[1], [1]], repeats=j, axis=1)
        generator_matrix = np.hstack((H, I, J))
        super().__init__(generator_matrix=generator_matrix)
        self._minimum_distance = h + i

    def __repr__(self):
        args = "{}".format(self._length)
        return "{}({})".format(self.__class__.__name__, args)

    def _default_decoder(self, dtype):
        if dtype == int:
            return "exhaustive_search_hard"
        elif dtype == float:
            return "exhaustive_search_soft"
