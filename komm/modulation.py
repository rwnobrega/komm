"""
Modulation
==========

Modulation schemes.

Real modulation
---------------

    RealModulation
    PAModulation

Complex modulation
------------------

    ComplexModulation
    ASKModulation
    PSKModulation
    QAModulation
"""

import itertools

import numpy as np

from .util import int2binlist

__all__ = ['RealModulation', 'PAModulation',
           'ComplexModulation', 'ASKModulation', 'PSKModulation', 'QAModulation']


class Modulation:
    """
    Modulation.
    """
    def __init__(self, constellation, labeling='natural'):
        self._init_constellation(constellation)
        self._init_labeling(labeling)
        self._channel_snr = 1.0

        self._hard_symbol_demodulator = \
            lambda recv: minimum_distance_demodulator(recv, self._constellation)
        self._soft_bit_demodulator = \
            lambda recv: soft_bit_demodulator(recv, self._constellation, self._inverse_mapping, self.channel_N0)

    def __repr__(self):
        args = 'constellation={}, labeling={}'.format(self._constellation.tolist(), self._labeling.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

    def _init_constellation(self, constellation):
        self._constellation = np.array(constellation)
        self._order = self._constellation.size
        if self._order & (self._order - 1):
            raise ValueError("The length of constellation must be a power of two")
        self._bits_per_symbol = (self._order - 1).bit_length()

    def _init_labeling(self, labeling):
        if labeling == 'natural':
            labeling = np.arange(self._order)
        elif labeling == 'reflected':
            labeling = np.arange(self._order)
            labeling ^= (labeling >> 1)
        elif labeling == 'reflected_2d':
            order_1d = int(np.sqrt(self._order))
            labeling_1d = np.arange(order_1d)
            labeling_1d ^= (labeling_1d >> 1)
            labeling = np.empty(self._order, dtype=np.int)
            for i, (i0, i1) in enumerate(itertools.product(labeling_1d, repeat=2)):
                labeling[i] = i0 + order_1d * i1

        self._labeling = labeling
        if not np.array_equal(np.sort(self._labeling), np.arange(self._order, dtype=np.int)):
            raise ValueError("The labeling must be a permutation of [0, ..., order)")
        self._mapping = {symbol: tuple(int2binlist(label, width=self._bits_per_symbol)) \
                         for (symbol, label) in enumerate(self._labeling)}
        self._inverse_mapping = dict((value, key) for key, value in self._mapping.items())

    @property
    def constellation(self):
        """
        The modulation constellation :math:`\\mathcal{S}`.
        """
        return self._constellation

    @property
    def labeling(self):
        """
        The modulation binary labeling :math:`\\mathcal{Q}`.
        """
        return self._labeling

    @property
    def order(self):
        """
        The modulation order :math:`M`.
        """
        return self._order

    @property
    def bits_per_symbol(self):
        """
        :math:`\log_2(M)`.
        """
        return self._bits_per_symbol

    @property
    def energy_per_symbol(self):
        """
        Mean symbol energy, supposing equiprobable symbols.
        """
        return np.real(self._constellation @ self._constellation.conj()) / self._order

    @property
    def energy_per_bit(self):
        """
        Mean bit energy, supposing equiprobable bits.
        """
        return self.energy_per_symbol / np.log2(self._order)

    def minimum_distance(self):
        """
        Constellation minimum euclidean distance.
        """
        pass

    @property
    def channel_snr(self):
        return self._channel_snr

    @property
    def channel_N0(self):
        return self.energy_per_symbol / self._channel_snr

    @channel_snr.setter
    def channel_snr(self, value):
        self._channel_snr = value

    def bits_to_symbols(self, bits):
        """
        Convert bits to symbols using the modulation labeling
        """
        m = self._bits_per_symbol
        n_symbols = len(bits) // m
        assert len(bits) == n_symbols * m
        symbols = np.empty(n_symbols, dtype=np.int)
        for i, bit_sequence in enumerate(np.reshape(bits, newshape=(n_symbols, m))):
            symbols[i] = self._inverse_mapping[tuple(bit_sequence)]
        return symbols

    def symbols_to_bits(self, symbols):
        """
        Convert symbols to bits using the modulation labeling
        """
        m = self._bits_per_symbol
        n_bits = len(symbols) * m
        bits = np.empty(n_bits, dtype=np.int)
        for i, symbol in enumerate(symbols):
            bits[i*m : (i + 1)*m] = self._mapping[symbol]
        return bits

    def modulate(self, bits):
        """
        Modulate bits.
        """
        symbols = self.bits_to_symbols(bits)
        return self._constellation[symbols]

    def demodulate(self, recv, decision_method='hard'):
        """
        Demodulate.
        """
        if decision_method == 'hard':
            symbols_hat = self._hard_symbol_demodulator(recv)
            return self.symbols_to_bits(symbols_hat)
        elif decision_method == 'soft':
            return self._soft_bit_demodulator(recv)
        else:
            raise ValueError("Parameter 'decision_method' should be either 'hard' of 'soft'")


class RealModulation(Modulation):
    """
    General real modulation scheme.

    A *real-valued modulation scheme* of order :math:`M` is defined by:

    - A *constellation* :math:`\\mathcal{S}`, which is an ordered subset of real numbers,
      with :math:`|\\mathcal{S}| = M`.
    - A *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of
      :math:`\\{ 0, 1, \ldots, M - 1 \\}`

    The order :math:`M` of the modulation must be a power of :math:`2`.

    Parameters
    ==========
    constellation : `array_like` of `int`
        Modulation constellation.
    labeling : `array_like` of `int`, or `str`, optional
        Modulation binary labeling, a permutation of range(M). Default is :code:`'natural'`.

    Examples
    ========
    >>> mod = komm.RealModulation(constellation=[-0.5, 0, 0.5, 2], labeling=[0, 1, 3, 2])
    >>> mod
    RealModulation(constellation=[-0.5, 0.0, 0.5, 2.0])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([-0.5,  0.5, -0.5,  0. ,  0. ])

    See also
    ========
    PAModulation, ComplexModulation
    """
    def __init__(self, constellation, labeling='natural'):
        super().__init__(np.array(constellation, dtype=np.float), labeling)


class PAModulation(RealModulation):
    """
    Pulse-amplitude modulation (PAM).

    A *pulse-amplitude modulation* (*PAM*) scheme of order :math:`M` is a real-valued modulation
    scheme (:class:`RealModulation`) with *uniformly spaced* constellation points. There are two
    options for the constellation :math:`\\mathcal{S}`:

    - *Single polarity*, in which :math:`\\mathcal{S} = \{{ 0, A, \ldots, (M-1)A \}}`.
    - *Double polarity*, in which :math:`\\mathcal{S} = \{{ \\pm A, \\pm 3A, \ldots, \\pm (M-1)A \}}`.

    The parameter :math:`A` is called the *base amplitude*.

    Parameters
    ==========
    order : `int`
        Modulation order :math:`M`.
        Must be a power of :math:`2`.
    polarity : `str`, optional
        Constellation polarity, either :code:`'single'` or :code:`'double'`.
        Default is :code:`'double'`.
    base_amplitude : `float`, optional
        Modulation base amplitude :math:`A`.
        Default is :code:`1.0`.
    labeling : `array_like` of `int`, or `str`, optional
        Modulation binary labeling, a permutation of :code:`range(M)`.
        Default is :code:`'reflected'`.

    Examples
    ========
    >>> pam = komm.PAModulation(order=4, polarity='single', base_amplitude=2.0)
    >>> pam
    PAModulation(order=4, polarity='single', base_amplitude=2.0)
    >>> pam.constellation
    array([ 0.,  2.,  4.,  6.])
    >>> pam.modulate([0, 0, 1, 0, 1, 1, 0, 1])
    array([ 0.,  2.,  4.,  6.])
    >>> pam.demodulate([0.99, 1.01, 4.99, 5.01])
    array([0, 0, 1, 0, 1, 1, 0, 1])

    See also
    ========
    RealModulation, ASKModulation
    """
    def __init__(self, order, polarity='double', base_amplitude=1.0, labeling='reflected'):
        if polarity == 'double':
            constellation = base_amplitude * np.linspace(-order + 1, order - 1, num=order, dtype=np.float)
        elif polarity == 'single':
            constellation = base_amplitude * np.linspace(0, order - 1, num=order, dtype=np.float)
        else:
            raise ValueError("Parameter 'polarity' should be either 'double' of 'single'")

        super().__init__(constellation, labeling)

        self._polarity = polarity
        self._base_amplitude = base_amplitude

        self._hard_symbol_demodulator = \
            lambda recv: uniform_real_hard_demodulator(np.array(recv) / base_amplitude, self.order)

        if order == 2:
            self._soft_bit_demodulator = \
            lambda recv: uniform_real_soft_bit_demodulator(np.array(recv) / base_amplitude, self._channel_snr)

    def __repr__(self):
        args = '{}, polarity="{}", base_amplitude={}'.format(self._order, self._polarity, self._base_amplitude)
        return '{}({})'.format(self.__class__.__name__, args)


class ComplexModulation(Modulation):
    """
    General complex modulation scheme.

    A *complex-valued modulation scheme* of order :math:`M` is defined by:

    - A *constellation* :math:`\\mathcal{S}`, which is an ordered subset of complex numbers,
      with :math:`|\\mathcal{S}| = M`.
    - A *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of
      :math:`\\{ 0, 1, \ldots, M - 1 \\}`

    The order :math:`M` of the modulation must be a power of :math:`2`.

    Parameters
    ==========
    constellation : `array_like` of `int`
        Modulation constellation.
    labeling : `array_like` of `int`, or `str`, optional
        Modulation binary labeling, a permutation of range(M). Default is :code:`'natural'`.

    Examples
    ========
    >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
    >>> mod
    ComplexModulation(constellation=[0j, (-1+0j), (1+0j), 1j])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0])
    >>> array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j])

    See also
    ========
    ASKModulation, PSKModulation, QAModulation, RealModulation
    """
    def __init__(self, constellation, labeling='natural'):
        super().__init__(np.array(constellation, dtype=np.complex), labeling)


class ASKModulation(ComplexModulation):
    """
    Amplitude-shift keying (ASK) modulation.

    An *amplitude-shift keying* (*ASK*) modulation scheme of order :math:`M` is a complex-valued
    modulation scheme (:class:`ComplexModulation`) with *uniformly spaced* constellation points.
    There are two options for the constellation :math:`\\mathcal{S}`:

    - *Single polarity*, in which :math:`\\mathcal{S} = \{{ 0, A, \ldots, (M-1)A \}}`.
    - *Double polarity*, in which :math:`\\mathcal{S} = \{{ \\pm A, \\pm 3A, \ldots, \\pm (M-1)A \}}`.

    Here, :math:`A` is set to :math:`1`.

    Parameters
    ==========
    order : `int`
        Modulation order. Must be a power of :math:`2`.
    polarity : `str`, optional
        Constellation polarity, either :code:`'single'` or :code:`'double'`. Default is :code:`'double'`.
    labeling : `array_like` of `int`, or `str`, optional
        Modulation binary labeling, a permutation of range(M). Default is :code:`'reflected'`.

    See also
    ========
    RealModulation, ASKModulation

    Parameters
    ==========
    labeling:
        Default is 'reflected' (Gray code)
    """
    def __init__(self, order, labeling='reflected'):
        constellation = np.linspace(-order + 1, order - 1, num=order, dtype=np.float)

        super().__init__(constellation, labeling)

        self._hard_symbol_demodulator = \
            lambda recv: ask_hard_demodulator(np.array(recv), self.order)

    def __repr__(self):
        args = '{}'.format(self._order)
        return '{}({})'.format(self.__class__.__name__, args)


class PSKModulation(ComplexModulation):
    """
    Phase-shift keying (PSK) modulation.

    Parameters
    ==========
    labeling:
        Default is 'reflected' (Gray code)
    """
    def __init__(self, order, labeling='reflected'):
        constellation = np.exp(2j*np.pi*np.arange(order) / order)

        super().__init__(constellation, labeling)

        self._hard_symbol_demodulator = \
            lambda recv: psk_hard_demodulator(recv, self._order)

        if order == 2:
            self._soft_bit_demodulator = \
                lambda recv: bpsk_soft_bit_demodulator(np.array(recv), self._channel_snr)
        elif order == 4 and labeling == 'reflected':
            self._soft_bit_demodulator = \
                lambda recv: qpsk_soft_bit_demodulator_reflected(np.array(recv), self._channel_snr)

    def __repr__(self):
        args = '{}'.format(self._order)
        return '{}({})'.format(self.__class__.__name__, args)


class QAModulation(ComplexModulation):
    """
    Quadratude-amplitude modulation (QAM).

    For even m, square
    For odd m, cross or rectangular

    .. image:: figure/qam16.png
       :alt: 16-QAM constellation
       :align: center

    Parameters
    ==========
    order:
        Here.
    labeling:
        Default is '2d_reflected' (Gray code)
    """
    def __init__(self, order, labeling='reflected_2d'):
        assert order in [2**(2*i) for i in range (1, 11)]

        order_1d = int(np.sqrt(order))
        constellation_1d = np.linspace(-order_1d + 1, order_1d - 1, num=order_1d, dtype=np.float)
        constellation = (constellation_1d + 1j*constellation_1d[np.newaxis].T).flatten()

        super().__init__(constellation, labeling)

        self._hard_symbol_demodulator = \
            lambda recv: rectangular_hard_demodulator(np.array(recv), self._order)

    def __repr__(self):
        args = '{}'.format(self._order)
        return '{}({})'.format(self.__class__.__name__, args)




def minimum_distance_demodulator(recv, constellation):
    """General minimum distance hard demodulator."""
    mpoints, mconst = np.meshgrid(recv, constellation)
    return np.argmin(np.absolute(mpoints - mconst), axis=0)


def soft_bit_demodulator(recv, constellation, inverse_mapping, N0):
    """Computes L-values of received points"""
    m = len(list(inverse_mapping.keys())[0])

    def pdf_recv_given_bit(bit_index, bit_value):
        bits = np.empty(m, dtype=np.int)
        bits[bit_index] = bit_value
        rest_index = np.setdiff1d(np.arange(m), [bit_index])
        f = 0.0
        for b_rest in itertools.product([0, 1], repeat=m-1):
            bits[rest_index] = b_rest
            point = constellation[inverse_mapping[tuple(bits)]]
            f += np.exp(-np.abs(recv - point)**2 / N0)
        return f

    soft_bits = np.empty(len(recv)*m, dtype=np.float)
    for bit_index in range(m):
        p0 = pdf_recv_given_bit(bit_index, 0)
        p1 = pdf_recv_given_bit(bit_index, 1)
        soft_bits[bit_index::m] = np.log(p0 / p1)

    return soft_bits


def uniform_real_hard_demodulator(recv, order):
    return np.clip(np.around((recv + order - 1) / 2), 0, order - 1).astype(np.int)


def uniform_real_soft_bit_demodulator(recv, snr):
    return -4 * snr * recv


def ask_hard_demodulator(recv, order):
    return np.clip(np.around((recv.real + order - 1) / 2), 0, order - 1).astype(np.int)


def psk_hard_demodulator(recv, order):
    phase_in_turns = np.angle(recv) / (2 * np.pi)
    return np.mod(np.around(phase_in_turns * order).astype(np.int), order)


def bpsk_soft_bit_demodulator(recv, snr):
    return 4 * snr * recv.real


def qpsk_soft_bit_demodulator_reflected(recv, snr):
    recv_rotated = recv * np.exp(2j * np.pi / 8)
    soft_bits = np.empty(2*recv.size, dtype=np.float)
    soft_bits[0::2] = np.sqrt(8) * snr * recv_rotated.real
    soft_bits[1::2] = np.sqrt(8) * snr * recv_rotated.imag
    return soft_bits


def rectangular_hard_demodulator(recv, order):
    L = int(np.sqrt(order))
    s_real = np.clip(np.around((recv.real + L - 1) / 2), 0, L - 1).astype(np.int)
    s_imag = np.clip(np.around((recv.imag + L - 1) / 2), 0, L - 1).astype(np.int)
    return s_real + L * s_imag
