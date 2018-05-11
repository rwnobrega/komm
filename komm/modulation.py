import itertools

import numpy as np

from .util import int2binlist

__all__ = ['RealModulation', 'PAModulation',
           'ComplexModulation', 'ASKModulation', 'PSKModulation', 'QAModulation']


class Modulation:
    """
    General modulation scheme.
    """
    def __init__(self, constellation, labeling='natural'):
        self._init_constellation(constellation)
        self._init_labeling(labeling)
        self._channel_snr = 1.0
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

        self._hard_symbol_demodulator = \
            lambda recv: minimum_distance_demodulator(recv, self._constellation)
        self._soft_bit_demodulator = \
            lambda recv: soft_bit_demodulator(recv, self._constellation, self._inverse_mapping, self._channel_N0)

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
        The constellation :math:`\\mathcal{S}` of the modulation. This property is read-only.
        """
        return self._constellation

    @property
    def labeling(self):
        """
        The binary labeling :math:`\\mathcal{Q}` of the modulation. This property is read-only.
        """
        return self._labeling

    @property
    def order(self):
        """
        The order :math:`M` of the modulation. This property is read-only.
        """
        return self._order

    @property
    def bits_per_symbol(self):
        """
        The number of bits per symbol of the modulation. It is given by :math:`\log_2(M)`, where :math:`M` is the order of the modulation. This property is read-only.
        """
        return self._bits_per_symbol

    @property
    def energy_per_symbol(self):
        """
        The average symbol energy :math:`E_\\mathrm{s}` of the constellation. It assumes equiprobable symbols. It is given by

        .. math::

            E_\\mathrm{s} = \\sum_{s \\in \\mathcal{S}} \\frac{1}{M} \\| s \\|^2,

        where :math:`\| s \|^2` is the energy of symbol :math:`s \\in \\mathcal{S}` and :math:`M` is the order of the modulation. This property is read-only.
        """
        return np.real(np.dot(self._constellation, self._constellation.conj())) / self._order

    @property
    def energy_per_bit(self):
        """
        The average bit energy :math:`E_\\mathrm{b}` of the constellation. It assumes equiprobable symbols. It is given by :math:`E_\\mathrm{b} = `E_\\mathrm{s} / \\log_2(M)`, where :math:`E_\\mathrm{s}` is the average symbol energy, and :math:`M` is the order of the modulation.
        """
        return self.energy_per_symbol / np.log2(self._order)

    @property
    def minimum_distance(self):
        """
        The minimum euclidean distance of the constellation.
        """
        pass

    @property
    def channel_snr(self):
        """
        The signal-to-noise radio :math:`\\mathrm{SNR}` of the channel. This is used in soft-decision methods. This is a read-and-write property.
        """
        return self._channel_snr

    @channel_snr.setter
    def channel_snr(self, value):
        self._channel_snr = value
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

    def bits_to_symbols(self, bits):
        """
        Convert bits to symbols using the modulation labeling.

        **Input:**

        :code:`bits` : 1D-array of :obj:`int`
            The bits to be converted. It should be a 1D-array of integers in the set :math:`\\{ 0, 1 \}}`. Its length must be a multiple of :math:`\\log_2 M`.

        **Output:**

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols corresponding to :code:`bits`. It is a 1D-array of integers in the set :math:`[O : M)`. Its length is equal to the length of :code:`bits` divided by :math:`\\log_2 M`.
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
        Convert symbols to bits using the modulation labeling.

        TODO: Complete me.

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
    General real modulation scheme. A *real-valued modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of real numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.

    .. rubric:: Examples

    >>> mod = komm.RealModulation(constellation=[-0.5, 0, 0.5, 2], labeling=[0, 1, 3, 2])
    >>> mod
    RealModulation(constellation=[-0.5, 0.0, 0.5, 2.0])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([-0.5,  0.5, -0.5,  0. ,  0. ])
    """
    def __init__(self, constellation, labeling='natural'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`float`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array of :math:`M` floats.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as an 1D-array of integers, in which case must be permutation of :math:`[0:M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'natural'`.
        """
        if isinstance(labeling, str):
            if labeling not in ['natural', 'reflected']:
                raise ValueError("Only 'natural' or 'reflected' are supported for RealModulation")
        super().__init__(np.array(constellation, dtype=np.float), labeling)


class PAModulation(RealModulation):
    """
    Pulse-amplitude modulation (PAM). A *pulse-amplitude modulation* (*PAM*) scheme of order :math:`M` is a real-valued modulation scheme (:class:`RealModulation`) with *uniformly spaced* constellation points. There are two options for the constellation :math:`\\mathcal{S}`:

    - *Single polarity*, in which :math:`\\mathcal{S} = \{{ 0, A, \ldots, (M-1)A \}}`.
    - *Double polarity*, in which :math:`\\mathcal{S} = \{{ \\pm A, \\pm 3A, \ldots, \\pm (M-1)A \}}`.

    The parameter :math:`A` is called the *base amplitude*.

    .. rubric:: Examples

    >>> pam = komm.PAModulation(order=4, polarity='single', base_amplitude=2.0)
    >>> pam
    PAModulation(order=4, polarity='single', base_amplitude=2.0)
    >>> pam.constellation
    array([ 0.,  2.,  4.,  6.])
    >>> pam.modulate([0, 0, 1, 0, 1, 1, 0, 1])
    array([ 0.,  2.,  4.,  6.])
    >>> pam.demodulate([0.99, 1.01, 4.99, 5.01])
    array([0, 0, 1, 0, 1, 1, 0, 1])
    """
    def __init__(self, order, polarity='double', base_amplitude=1.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`polarity` : :obj:`str`, optional
            The polarity of the constellation. Should be either :code:`'single'` or :code:`'double'`. The default value is :code:`'double'`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as an 1D-array of integers, in which case must be permutation of :math:`[0:M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).
        """
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
        args = "{}, polarity='{}', base_amplitude={}".format(self._order, self._polarity, self._base_amplitude)
        return '{}({})'.format(self.__class__.__name__, args)


class ComplexModulation(Modulation):
    """
    General complex modulation scheme. A *complex-valued modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of complex numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.

    .. rubric:: Examples

    >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
    >>> mod
    ComplexModulation(constellation=[0j, (-1+0j), (1+0j), 1j])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0])
    >>> array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j])
    """
    def __init__(self, constellation, labeling='natural'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`complex`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array of :math:`M` complex numbers.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as an 1D-array of integers, in which case must be permutation of :math:`[0:M)`, or as a string, in which case must be one of :code:`'natural'`, :code:`'reflected'`, or :code:`'reflected_2d'` The default value is :code:`'natural'`.
        """
        if isinstance(labeling, str):
            if labeling not in ['natural', 'reflected', 'reflected_2d']:
                raise ValueError("Only 'natural', 'reflected', or 'reflected_2d' are supported for ComplexModulation")

        super().__init__(np.array(constellation, dtype=np.complex), labeling)


class ASKModulation(ComplexModulation):
    """
    Amplitude-shift keying (ASK) modulation. An *amplitude-shift keying* (*ASK*) modulation scheme of order :math:`M` is a complex-valued modulation scheme (:class:`ComplexModulation`) with *uniformly spaced* constellation points. The constellation is given by :math:`\\mathcal{S} = \{{ 0, A, \ldots, (M-1)A \}}`, where :math:`A` is called the *base amplitude*.

    .. rubric:: Examples

    >>> ask = komm.ASKModulation(order=4, base_amplitude=2.0)
    >>> ask
    ASKModulation(4, base_amplitude=2.0)
    >>> ask.constellation
    array([0.+0.j, 2.+0.j, 4.+0.j, 6.+0.j])
    >>> ask.modulate([0, 0, 1, 0, 1, 1, 0, 1])
    array([0.+0.j, 2.+0.j, 4.+0.j, 6.+0.j])
    >>> ask.demodulate([(0.99+0.3j), (1.01-0.5j), (4.99+0.7j), (5.01-0.9j)])
    array([0, 0, 1, 0, 1, 1, 0, 1])
    """
    def __init__(self, order, base_amplitude=1.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as an 1D-array of integers, in which case must be permutation of :math:`[0:M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).
        """
        constellation = base_amplitude * np.linspace(0, order - 1, num=order, dtype=np.float)

        if isinstance(labeling, str):
            if labeling not in ['natural', 'reflected']:
                raise ValueError("Only 'natural' or 'reflected' are supported for ASKModulation")

        super().__init__(constellation, labeling)

        self._base_amplitude = base_amplitude

        self._hard_symbol_demodulator = \
            lambda recv: ask_hard_demodulator(np.array(recv) / base_amplitude, self.order)

    def __repr__(self):
        args = '{}, base_amplitude={}'.format(self._order, self._base_amplitude)
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
