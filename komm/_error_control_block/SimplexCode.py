from .BlockCode import BlockCode
from .util import _hamming_parity_submatrix


class SimplexCode(BlockCode):
    """
    Simplex (maximum-length) code. For a given dimension :math:`k`, it is the linear block code (:class:`BlockCode`) with generator matrix whose columns are all the :math:`2^k - 1` nonzero binary :math:`k`-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: :math:`n = 2^k - 1`
    - Dimension: :math:`k`
    - Redundancy: :math:`m = 2^k - k - 1`
    - Minimum distance: :math:`d = 2^{k - 1}`

    This class constructs the code in systematic form, with the information set on the left.

    .. rubric:: Decoding methods

    [[decoding_methods]]

    .. rubric:: Notes

    - For :math:`k = 2` it reduces to the single parity check code (:class:`SingleParityCheckCode`) of length :math:`3`.
    - Its dual is the Hamming code (:class:`HammingCode`).
    - Simplex codes are constant-weight codes.

    .. rubric:: Examples

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
    def __init__(self, k):
        """
        Constructor for the class. It expects the following parameter:

        :code:`k` : :obj:`int`
            The dimension :math:`k` of the code. Must satisfy :math:`k \\geq 2`.
        """
        P = _hamming_parity_submatrix(k).T
        super().__init__(parity_submatrix=P)
        self._minimum_distance = 2**(k - 1)
        self._k = k

    def __repr__(self):
        args = '{}'.format(self._k)
        return '{}({})'.format(self.__class__.__name__, args)
