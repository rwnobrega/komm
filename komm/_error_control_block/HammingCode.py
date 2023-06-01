from .BlockCode import BlockCode
from .util import _extended_parity_submatrix, _hamming_parity_submatrix


class HammingCode(BlockCode):
    """
    Hamming code. For a given redundancy :math:`m`, it is the linear block code (:class:`BlockCode`) with parity-check matrix whose columns are all the :math:`2^m - 1` nonzero binary :math:`m`-tuples. The Hamming code has the following parameters:

    - Length: :math:`n = 2^m - 1`
    - Redundancy: :math:`m`
    - Dimension: :math:`k = 2^m - m - 1`
    - Minimum distance: :math:`d = 3`

    This class constructs the code in systematic form, with the information set on the left.

    References: :cite:`Lin.Costello.04` (Sec 4.1)

    .. rubric:: Decoding methods

    [[decoding_methods]]

    .. rubric:: Notes

    - For :math:`m = 2` it reduces to the repetition code (:class:`RepetitionCode`) of length :math:`3`.
    - Its dual is the simplex code (:class:`SimplexCode`).
    - Hamming codes are perfect codes.

    .. rubric:: Examples

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
        """
        Constructor for the class. It expects the following parameters:

        :code:`m` : :obj:`int`
            The redundancy :math:`m` of the code. Must satisfy :math:`m \\geq 2`.

        :code:`extended` : :obj:`bool`, optional
            If :code:`True`, constructs the code in extended version. The default value is :code:`False`.
        """
        P = _hamming_parity_submatrix(m)
        if extended:
            P = _extended_parity_submatrix(P)
        super().__init__(parity_submatrix=P)
        self._minimum_distance = 4 if extended else 3
        self._m = m
        self._extended = extended

    def __repr__(self):
        args = "{}".format(self._m)
        if self._extended:
            args += ", extended=True"
        return "{}({})".format(self.__class__.__name__, args)
