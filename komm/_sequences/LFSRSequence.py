import numpy as np
from .._algebra import BinaryPolynomial
from .BinarySequence import BinarySequence


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
        code = np.empty(L, dtype=int)
        for i in range(L):
            code[i] = state[-1]
            state[-1] = np.count_nonzero(state[taps - 1]) % 2
            state = np.roll(state, 1)
        return code
