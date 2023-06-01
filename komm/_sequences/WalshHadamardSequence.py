import numpy as np

from .._util import binlist2int, int2binlist
from .BinarySequence import BinarySequence


class WalshHadamardSequence(BinarySequence):
    """
    Walsh--Hadamard sequence. Consider the following recursive matrix construction:

    .. math::

        H_{1} =
        \\begin{bmatrix}
            +1
        \\end{bmatrix}, \\qquad
        H_{2^n} =
        \\begin{bmatrix}
            H_{2^{n-1}} & H_{2^{n-1}} \\\\
            H_{2^{n-1}} & -H_{2^{n-1}}
        \\end{bmatrix},

    for :math:`n = 1, 2, \\ldots`. For example, for :math:`n = 3`,

    .. math::

        H_{8} =
        \\begin{bmatrix}
            +1 & +1 & +1 & +1 & +1 & +1 & +1 & +1 \\\\
            +1 & -1 & +1 & -1 & +1 & -1 & +1 & -1 \\\\
            +1 & +1 & -1 & -1 & +1 & +1 & -1 & -1 \\\\
            +1 & -1 & -1 & +1 & +1 & -1 & -1 & +1 \\\\
            +1 & +1 & +1 & +1 & -1 & -1 & -1 & -1 \\\\
            +1 & -1 & +1 & -1 & -1 & +1 & -1 & +1 \\\\
            +1 & +1 & -1 & -1 & -1 & -1 & +1 & +1 \\\\
            +1 & -1 & -1 & +1 & -1 & +1 & +1 & -1 \\\\
        \\end{bmatrix}

    The above matrix is said to be in *natural ordering*. If the rows of the matrix are rearranged by first applying the bit-reversal permutation and then the Gray-code permutation, the following matrix is obtained:

    .. math::
        H_{8}^{\\mathrm{s}} =
        \\begin{bmatrix}
            +1 & +1 & +1 & +1 & +1 & +1 & +1 & +1 \\\\
            +1 & +1 & +1 & +1 & -1 & -1 & -1 & -1 \\\\
            +1 & +1 & -1 & -1 & -1 & -1 & +1 & +1 \\\\
            +1 & +1 & -1 & -1 & +1 & +1 & -1 & -1 \\\\
            +1 & -1 & -1 & +1 & +1 & -1 & -1 & +1 \\\\
            +1 & -1 & -1 & +1 & -1 & +1 & +1 & -1 \\\\
            +1 & -1 & +1 & -1 & -1 & +1 & -1 & +1 \\\\
            +1 & -1 & +1 & -1 & +1 & -1 & +1 & -1 \\\\
        \\end{bmatrix}

    The above matrix is said to be in *sequency ordering*. It has the property that row :math:`i` has exactly :math:`i` signal changes.

    The Walsh--Hadamard sequence of *length* :math:`L` and *index* :math:`i \\in [0 : L)` is the binary sequence (:obj:`BinarySequence`) whose polar format is the :math:`i`-th row of :math:`H_L`, if assuming natural ordering, or :math:`H_L^{\\mathrm{s}}`, if assuming sequency ordering.

    [1] https://en.wikipedia.org/wiki/Hadamard_matrix; [2] https://en.wikipedia.org/wiki/Walsh_matrix
    """

    def __init__(self, length, ordering="natural", index=0):
        """
        Constructor for the class. It expects the following parameters:

        :code:`length` : :obj:`int`
            Length :math:`L` of the Walsh--Hadamard sequence. Must be a power of two.

        :code:`ordering` : :obj:`str`, optional
            Ordering to be assumed. Should be one of :code:`'natural'`, :code:`'sequency'`, or :code:`'dyadic'`. The default value is :code:`'natural'`.

        :code:`index` : :obj:`int`, optional
            Index of the Walsh--Hadamard sequence, with respect to the ordering assumed. Must be in the set :math:`[0 : L)`. The default value is :code:`0`.

        .. rubric:: Examples

        >>> walsh_hadamard = komm.WalshHadamardSequence(length=64, ordering='sequency', index=60)
        >>> walsh_hadamard.polar_sequence[:16]
        array([ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1])

        >>> walsh_hadamard = komm.WalshHadamardSequence(length=128, ordering='natural', index=60)
        >>> walsh_hadamard.polar_sequence[:16]
        array([ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1])
        """
        if length & (length - 1):
            raise ValueError("The length of sequence must be a power of two")

        if not 0 <= index < length:
            raise ValueError("Parameter 'index' must be in [0, ..., length)")

        if ordering == "natural":
            natural_index = index
        elif ordering == "sequency":
            width = (length - 1).bit_length()
            index_gray = index ^ (index >> 1)
            natural_index = binlist2int(reversed(int2binlist(index_gray, width)))
        elif ordering == "dyadic":
            raise NotImplementedError
        else:
            raise ValueError("Parameter 'ordering' must be 'natural', 'sequency' or 'dyadic'")

        self._index = index
        self._ordering = ordering
        super().__init__(polar_sequence=self._hadamard_matrix(length)[natural_index])

    def __repr__(self):
        args = "length={}, ordering='{}', index={}".format(self._length, self._ordering, self._index)
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def index(self):
        """
        The index of the Walsh--Hadamard sequence, with respect to the ordering assumed.
        """
        return self._index

    @property
    def ordering(self):
        """
        The ordering assumed.
        """
        return self._ordering

    @staticmethod
    def _hadamard_matrix(length):
        h = np.array([[1]])
        g = np.array([[1, 1], [1, -1]])
        for _ in range(length.bit_length() - 1):
            h = np.kron(h, g)
        return h
