"""
Sequences
=========

Sequences...

Binary sequences
----------------

    BarkerSequence
    WalshHadamardSequence
    LFSRSequence
"""

import numpy as np

from .algebra import \
    BinaryPolynomial

from .util import \
    int2binlist, binlist2int

__all__ = ['Sequence',
           'BarkerSequence', 'WalshHadamardSequence', 'LFSRSequence']


class Sequence:
    """
    Sequence.
    """
    def __init__(self, sequence):
        self._sequence = np.array(sequence)
        self._length = self._sequence.size

    def __repr__(self):
        return f'{self.__class__.__name__}(sequence={self._sequence.tolist()})'

    @property
    def sequence(self):
        return self._sequence

    @property
    def length(self):
        return self._length


class BarkerSequence(Sequence):
    """
    Barker sequence.

    ======  =========================
    Length  Barker sequence
    ======  =========================
    2       1 0
    3       1 1 0
    4       1 1 0 1
    5       1 1 1 0 1
    7       1 1 1 0 0 1 0
    11      1 1 1 0 0 0 1 0 0 1 0
    13      1 1 1 1 1 0 0 1 1 0 1 0 1
    ======  =========================

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Barker_code
    """
    def __init__(self, length):
        super().__init__(self._barker_sequence(length))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.length})'

    @staticmethod
    def _barker_sequence(length):
        DICT = {
            2: [1, 0],
            3: [1, 1, 0],
            4: [1, 1, 0, 1],
            5: [1, 1, 1, 0, 1],
            7: [1, 1, 1, 0, 0, 1, 0],
            11: [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            13: [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        }
        return DICT[length]


class WalshHadamardSequence(Sequence):
    """
    Walsh--Hadamard sequence.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Hadamard_matrix
    [2] https://en.wikipedia.org/wiki/Walsh_matrix
    """
    def __init__(self, length, ordering='natural', index=0):
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
        super().__init__(self._hadamard_matrix(length)[natural_index])

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(length={self._length}, ordering='{self._ordering}', index={self._index})"

    @property
    def index(self):
        return self._index

    @staticmethod
    def _hadamard_matrix(length):
        h = np.array([[1]])
        g = np.array([[1, 1], [1, -1]])
        for _ in range(length.bit_length() - 1):
            h = np.kron(h, g)
        return (h < 0).astype(np.int)


class LFSRSequence(Sequence):
    """
    Linear-feedback shift register (LFSR).

    Some polynomials for maximum-length sequences (MLS):

    ================  =======================
    Number of states  Feedback polynomial
    ================  =======================
    2                 :code:`0b111`
    3                 :code:`0b1101`
    4                 :code:`0b11001`
    5                 :code:`0b101001`
    6                 :code:`0b1100001`
    7                 :code:`0b11000001`
    8                 :code:`0b101110001`
    9                 :code:`0b1000100001`
    10                :code:`0b10010000001`
    11                :code:`0b101000000001`
    12                :code:`0b1110000010001`
    ================  =======================

    .. rubric:: Parameters

        feedback_poly
            Feedback connections.
        start_state_poly
            Initial state.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Linear-feedback_shift_register
    [2] https://en.wikipedia.org/wiki/Maximum_length_sequence
    """
    def __init__(self, feedback_poly, start_state_poly=0b1):
        self._feedback_poly = BinaryPolynomial(feedback_poly)
        self._start_state_poly = BinaryPolynomial(start_state_poly)
        super().__init__(self._lfsr_sequence(self._feedback_poly, self._start_state_poly))

    @classmethod
    def maximum_length_sequence(cls, num_states):
        """
        Constructor.
        """
        DICT = {
            2: 0b111,
            3: 0b1101,
            4: 0b11001,
            5: 0b101001,
            6: 0b1100001,
            7: 0b11000001,
            8: 0b101110001,
            9: 0b1000100001,
            10: 0b10010000001,
            11: 0b101000000001,
            12: 0b1110000010001,
        }
        return cls(DICT[num_states])

    def __repr__(self):
        args = f'feedback_poly={self._feedback_poly}, start_state_poly={self._start_state_poly}'
        return f'{self.__class__.__name__}({args})'

    @property
    def feedback_poly(self):
        return self._feedback_poly

    @property
    def start_state_poly(self):
        return self._start_state_poly\

    @staticmethod
    def _lfsr_sequence(feedback_poly, start_state_poly):
        taps = (feedback_poly + BinaryPolynomial(1)).exponents()
        start_state = start_state_poly.coefficients(width=feedback_poly.degree)
        m = taps[-1]
        L = 2**m - 1
        state = np.copy(start_state)
        code = np.empty(L, dtype=np.int)
        for i in range(L):
            code[i] = state[-1]
            state[-1] = np.bitwise_xor.reduce(state[taps - 1])
            state = np.roll(state, 1)
        return code
