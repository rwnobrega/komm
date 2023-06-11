import numpy as np
from scipy import special

from .._aux import tag
from .BlockCode import BlockCode


class SingleParityCheckCode(BlockCode):
    r"""
    Single parity check code. For a given length $n$, it is the [linear block code](/ref/BlockCode) whose codewords are obtained by extending $n - 1$ information bits with a single parity-check bit. The repetition code has the following parameters:

    - Length: $n$.
    - Dimension: $k = n - 1$.
    - Minimum distance: $d = 2$.

    .. rubric:: Decoding methods

    [[decoding_methods]]

    Notes:

        - Its dual is the [repetition code](/ref/RepetitionCode).

    Examples:

        >>> code = komm.SingleParityCheckCode(5)
        >>> (code.length, code.dimension, code.minimum_distance)
        (5, 4, 2)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1],
               [0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1],
               [0, 0, 0, 1, 1]])
        >>> code.parity_check_matrix
        array([[1, 1, 1, 1, 1]])
        >>> code.encode([1, 0, 1, 1])
        array([1, 0, 1, 1, 1])
    """

    def __init__(self, n):
        r"""
        Constructor for the class.

        Parameters:

            n (:obj:`int`): The length $n$ of the code. Must be a positive integer.
        """
        super().__init__(parity_submatrix=np.ones((1, n - 1), dtype=int).T)
        self._minimum_distance = 2
        self._codeword_weight_distribution = np.zeros(n + 1, dtype=int)
        for w in range(0, n + 1, 2):
            self._codeword_weight_distribution[w] = special.comb(n, w, exact=True)

    def __repr__(self):
        args = "{}".format(self._length)
        return "{}({})".format(self.__class__.__name__, args)

    @tag(name="Wagner", input_type="soft", target="codeword")
    def _decode_wagner(self, recvword):
        r"""
        Wagner decoder. A soft-decision decoder for SingleParityCheck codes only. See Costello, Forney: Channel Coding: The Road to Channel Capacity.
        """
        codeword_hat = recvword < 0
        if np.count_nonzero(codeword_hat) % 2 != 0:
            i = np.argmin(np.abs(recvword))
            codeword_hat[i] ^= 1
        return codeword_hat.astype(int)

    def _default_decoder(self, dtype):
        if dtype == float:
            return "wagner"
        else:
            return super()._default_decoder(dtype)
