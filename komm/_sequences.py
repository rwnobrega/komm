import functools

import numpy as np

from ._algebra import \
    BinaryPolynomial

from ._util import \
    int2binlist, binlist2int

__all__ = ['BinarySequence',
           'BarkerSequence', 'WalshHadamardSequence',
           'LFSRSequence', 'GoldSequence', 'KasamiSequence',
           'ZadoffChuSequence']


class BinarySequence:
    """
    General binary sequence. It may be represented either in *bit format*, denoted by :math:`b[n]`, with elements in the set :math:`\\{ 0, 1 \\}`, or in *polar format*, denoted by :math:`a[n]`, with elements in the set :math:`\\{ \\pm 1 \\}`. The correspondences :math:`0 \\mapsto +1` and :math:`1 \\mapsto -1` from bit format to polar format is assumed.
    """
    def __init__(self, **kwargs):
        """
        Constructor for the class. It expects *exactly one* the following parameters:

        :code:`bit_sequence` : 1D-array of :obj:`int`
            The binary sequence in bit format. Must be an 1D-array with elements in :math:`\\{ 0, 1 \\}`.

        :code:`polar_sequence` : 1D-array of :obj:`int`
            The binary sequence in polar format. Must be an 1D-array with elements in :math:`\\{ \\pm 1 \\}`.
        """
        kwargs_set = set(kwargs.keys())
        if kwargs_set == {'bit_sequence'}:
            self._bit_sequence = np.array(kwargs['bit_sequence'], dtype=np.int)
            self._polar_sequence = (-1)**self._bit_sequence
            self._constructed_from = 'bit_sequence'
        elif kwargs_set == {'polar_sequence'}:
            self._polar_sequence = np.array(kwargs['polar_sequence'], dtype=np.int)
            self._bit_sequence = 1 * (self._polar_sequence < 0)
            self._constructed_from = 'polar_sequence'
        else:
            raise ValueError("Either specify 'bit_sequence' or 'polar_sequence'")

        self._length = self._bit_sequence.size

    def __repr__(self):
        args = 'bit_sequence={}'.format(self._bit_sequence.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def bit_sequence(self):
        """
        The binary sequence in bit format, :math:`b[n] \\in \\{ 0, 1 \\}`.
        """
        return self._bit_sequence

    @property
    def polar_sequence(self):
        """
        The binary sequence in polar format, :math:`a[n] \\in \\{ \\pm 1 \\}`.
        """
        return self._polar_sequence

    @property
    def length(self):
        """
        The length (or period) :math:`L` of the binary sequence.
        """
        return self._length

    @property
    @functools.lru_cache()
    def _autocorrelation(self):
        seq = self._polar_sequence
        L = self._length
        return np.correlate(seq, seq, mode='full')[L - 1:]

    @property
    @functools.lru_cache()
    def _cyclic_autocorrelation(self):
        seq = self._polar_sequence
        L = self._length
        return np.array([np.dot(seq, np.roll(seq, ell)) for ell in range(L)])

    def autocorrelation(self, shifts=None, normalized=False):
        """
        Returns the autocorrelation :math:`R[\\ell]` of the binary sequence. This is defined as

        .. math::

            R[\\ell] = \\sum_{n \\in \\mathbb{Z}} a[n] a_{\\ell}[n],

        where :math:`a[n]` is the binary sequence in polar format, and :math:`a_{\\ell}[n] = a[n - \\ell]` is the sequence :math:`a[n]` shifted by :math:`\\ell` positions. The autocorrelation :math:`R[\\ell]` is even and satisfies :math:`R[\\ell] = 0` for :math:`|\\ell| \\geq L`, where :math:`L` is the length of the binary sequence.

        **Parameters:**

        :code:`shifts` : 1D-array of :obj:`int`, optional.
            An 1D array containing the values of :math:`\\ell` for which the autocorrelation will be computed. The default value is :code:`range(L)`, that is, :math:`[0 : L)`.

        :code:`normalized` : :obj:`boolean`, optional
            If :code:`True`, returns the autocorrelation divided by :math:`L`, where :math:`L` is the length of the binary sequence, so that :math:`R[0] = 1`. The default value is :code:`False`.
        """
        L = self._length
        shifts = np.arange(L) if shifts is None else np.array(shifts)
        autocorrelation = np.array([self._autocorrelation[abs(ell)] if abs(ell) < L else 0 for ell in shifts])
        if normalized:
            return autocorrelation / L
        else:
            return autocorrelation

    def cyclic_autocorrelation(self, shifts=None, normalized=False):
        """
        Returns the cyclic autocorrelation :math:`\\tilde{R}[\\ell]` of the binary sequence. This is defined as

        .. math::

            \\tilde{R}[\\ell] = \\sum_{n \\in [0:L)} a[n] \\tilde{a}_{\\ell}[n],

        where :math:`a[n]` is the binary sequence in polar format, and :math:`\\tilde{a}_{\\ell}[n]` is the sequence :math:`a[n]` cyclic-shifted by :math:`\\ell` positions. The cyclic autocorrelation :math:`\\tilde{R}[\\ell]` is even and periodic with period :math:`L`, where :math:`L` is the period of the binary sequence.

        **Parameters:**

        :code:`shifts` : 1D-array of :obj:`int`, optional.
            An 1D array containing the values of :math:`\\ell` for which the cyclic autocorrelation will be computed. The default value is :code:`range(L)`, that is, :math:`[0 : L)`.

        :code:`normalized` : :obj:`boolean`, optional
            If :code:`True`, returns the cyclic autocorrelation divided by :math:`L`, where :math:`L` is the length of the binary sequence, so that :math:`\\tilde{R}[0] = 1`. The default value is :code:`False`.
        """
        L = self._length
        shifts = np.arange(L) if shifts is None else np.array(shifts)
        cyclic_autocorrelation = self._cyclic_autocorrelation[shifts % L]
        if normalized:
            return cyclic_autocorrelation / L
        else:
            return cyclic_autocorrelation


class BarkerSequence(BinarySequence):
    """
    Barker sequence. A Barker sequence is a binary sequence (:obj:`BinarySequence`) with autocorrelation :math:`R[\\ell]` satisfying :math:`|R[\\ell]| \\leq 1`, for :math:`\\ell \\neq 0`. The only known Barker sequences (up to negation and reversion) are shown in the table below.

    ================  =============================
    Length :math:`L`  Barker sequence :math:`b[n]`
    ================  =============================
    :math:`2`         :math:`01` and :math:`00`
    :math:`3`         :math:`001`
    :math:`4`         :math:`0010` and :math:`0001`
    :math:`5`         :math:`00010`
    :math:`7`         :math:`0001101`
    :math:`11`        :math:`00011101101`
    :math:`13`        :math:`0000011001010`
    ================  =============================

    [1] https://en.wikipedia.org/wiki/Barker_code
    """
    def __init__(self, length):
        """
        Constructor for the class. It expects the following parameter:

        :code:`length` : :obj:`int`
            Length of the Barker sequence. Must be in the set :math:`\\{ 2, 3, 4, 5, 7, 11, 13 \\}`.

        .. rubric:: Examples

        >>> barker = komm.BarkerSequence(length=13)
        >>> barker.polar_sequence
        array([ 1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1])
        >>> barker.autocorrelation()
        array([13,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1])
        """
        super().__init__(bit_sequence=self._barker_sequence(length))

    def __repr__(self):
        args = 'length={}'.format(self.length)
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _barker_sequence(length):
        DICT = {
            2: [0, 1],
            3: [0, 0, 1],
            4: [0, 0, 1, 0],
            5: [0, 0, 0, 1, 0],
            7: [0, 0, 0, 1, 1, 0, 1],
            11: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            13: [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        }
        return DICT[length]


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
    def __init__(self, length, ordering='natural', index=0):
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

        if ordering == 'natural':
            natural_index = index
        elif ordering == 'sequency':
            width = (length - 1).bit_length()
            index_gray = index ^ (index >> 1)
            natural_index = binlist2int(reversed(int2binlist(index_gray, width)))
        elif ordering == 'dyadic':
            raise NotImplementedError
        else:
            raise ValueError("Parameter 'ordering' must be 'natural', 'sequency' or 'dyadic'")

        self._index = index
        self._ordering = ordering
        super().__init__(polar_sequence=self._hadamard_matrix(length)[natural_index])

    def __repr__(self):
        args = "length={}, ordering='{}', index={}".format(self._length, self._ordering, self._index)
        return '{}({})'.format(self.__class__.__name__, args)

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


class LFSRSequence(BinarySequence):
    """
    Linear-feedback shift register (LFSR) sequence. It is the binary sequence (:obj:`BinarySequence`) obtained from the output of a LFSR. The LFSR feedback taps are specified as a binary polynomial :math:`p(X)` of degree :math:`n`, called the *feedback polynomial*. More specifically: if bit :math:`i` of the LFSR is tapped, for :math:`i \\in [1 : n]`, then the coefficient of :math:`X^i` in :math:`p(X)` is :math:`1`; otherwise, it is :math:`0`; moreover, the coefficient of :math:`X^0` in :math:`p(X)` is always :math:`1`. For example, the feedback polynomial corresponding to the LFSR in the figure below is :math:`p(X) = X^5 + X^2 + 1`, whose integer representation is :code:`0b100101`.

    .. image:: figures/lfsr_5_2.png
       :alt: Linear-feedback shift register example.
       :align: center

    The start state of the machine is specified by the so called *start state polynomial*. More specifically, the coefficient of :math:`X^i` in the start state polynomial is equal to the initial value of bit :math:`i` of the LFSR.

    .. rubric:: Maximum-length sequences

    If the feedback polynomial :math:`p(X)` is primitive, then the corresponding LFSR sequence will be a *maximum-length sequence* (MLS). Such sequences have the following cyclic autocorrelation:

    .. math::

        R[\\ell] =
        \\begin{cases}
            L, & \\ell = 0, \\pm L, \\pm 2L, \\ldots, \\\\
            -1, & \\text{otherwise},
        \\end{cases}

    where :math:`L` is the length of the sequence. The constructor :func:`maximum_length_sequence` can be use to construct an MLS.

    [1] https://en.wikipedia.org/wiki/Linear-feedback_shift_register; [2] https://en.wikipedia.org/wiki/Maximum_length_sequence

    .. rubric:: Examples

    >>> lfsr = komm.LFSRSequence(feedback_polynomial=0b100101)
    >>> lfsr.bit_sequence  #doctest: +NORMALIZE_WHITESPACE
    array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    >>> lfsr.cyclic_autocorrelation()  #doctest: +NORMALIZE_WHITESPACE
    array([31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    """
    def __init__(self, feedback_polynomial, start_state_polynomial=0b1):
        """
        Constructor for the class. It expects the following parameters:

        :code:`feedback_polynomial` : :obj:`BinaryPolynomial` or :obj:`int`
            The feedback polynomial of the LFSR, specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former.

        :code:`start_state_polynomial` : :obj:`BinaryPolynomial` or :obj:`int`, optional.
            The start state polynomial of the LFSR, specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former. The default value is :code:`0b1`.
        """
        self._feedback_polynomial = BinaryPolynomial(feedback_polynomial)
        self._start_state_polynomial = BinaryPolynomial(start_state_polynomial)
        super().__init__(bit_sequence=self._lfsr_sequence(self._feedback_polynomial, self._start_state_polynomial))

    @classmethod
    def maximum_length_sequence(cls, degree):
        """
        Constructs a maximum-length sequences (MLS) of a given degree. It expects the following parameter:

        :code:`degree` : :obj:`int`
            The degree :math:`n` of the MLS. Only degrees in the range :math:`[1 : 16]` are implemented.

        The feedback polynomial :math:`p(X)` is chosen according to the following table of primitive polynomials.

        ================  ================================  ================  ================================
        Degree :math:`n`  Feedback polynomial :math:`p(X)`  Degree :math:`n`  Feedback polynomial :math:`p(X)`
        ================  ================================  ================  ================================
        :math:`1`         :code:`0b11`                      :math:`9`         :code:`0b1000010001`
        :math:`2`         :code:`0b111`                     :math:`10`        :code:`0b10000001001`
        :math:`3`         :code:`0b1011`                    :math:`11`        :code:`0b100000000101`
        :math:`4`         :code:`0b10011`                   :math:`12`        :code:`0b1000001010011`
        :math:`5`         :code:`0b100101`                  :math:`13`        :code:`0b10000000011011`
        :math:`6`         :code:`0b1000011`                 :math:`14`        :code:`0b100010001000011`
        :math:`7`         :code:`0b10001001`                :math:`15`        :code:`0b1000000000000011`
        :math:`8`         :code:`0b100011101`               :math:`16`        :code:`0b10001000000001011`
        ================  ================================  ================  ================================
        """
        PRIMITIVE_POLYNOMIALS = {
            1: 0b11,
            2: 0b111,
            3: 0b1011,
            4: 0b10011,
            5: 0b100101,
            6: 0b1000011,
            7: 0b10001001,
            8: 0b100011101,
            9: 0b1000010001,
            10: 0b10000001001,
            11: 0b100000000101,
            12: 0b1000001010011,
            13: 0b10000000011011,
            14: 0b100010001000011,
            15: 0b1000000000000011,
            16: 0b10001000000001011}
        return cls(PRIMITIVE_POLYNOMIALS[degree])

    def __repr__(self):
        args = 'feedback_polynomial={}'.format(self._feedback_polynomial)
        if self._start_state_polynomial != BinaryPolynomial(0b1):
            args += ', start_state_polynomial={}'.format(self._start_state_polynomial)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def feedback_polynomial(self):
        """
        The feedback polynomial :math:`p(X)` of the LFSR.
        """
        return self._feedback_polynomial

    @property
    def start_state_polynomial(self):
        """
        The start state polynomial of the LFSR.
        """
        return self._start_state_polynomial

    @staticmethod
    def _lfsr_sequence(feedback_polynomial, start_state_polynomial):
        taps = (feedback_polynomial + BinaryPolynomial(1)).exponents()
        start_state = start_state_polynomial.coefficients(width=feedback_polynomial.degree)
        m = taps[-1]
        L = 2**m - 1
        state = np.copy(start_state)
        code = np.empty(L, dtype=np.int)
        for i in range(L):
            code[i] = state[-1]
            state[-1] = np.count_nonzero(state[taps - 1]) % 2
            state = np.roll(state, 1)
        return code


class GoldSequence:
    """
    Gold sequence [Not implemented yet].
    """
    pass


class KasamiSequence:
    """
    Kasami sequence [Not implemented yet].
    """
    pass


class ZadoffChuSequence:
    """
    Zadoff–Chu sequence [Not implemented yet].
    """
    pass
