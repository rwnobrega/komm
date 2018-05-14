import itertools

import numpy as np

from .util import int2binlist

__all__ = ['RealModulation', 'PAModulation',
           'ComplexModulation', 'ASKModulation', 'PSKModulation', 'APSKModulation', 'QAModulation']


class _Modulation:
    def __init__(self, constellation, labeling):
        self._constellation = np.array(constellation)
        self._order = self._constellation.size
        if self._order & (self._order - 1):
            raise ValueError("The length of constellation must be a power of two")
        self._bits_per_symbol = (self._order - 1).bit_length()

        self._labeling = np.array(labeling, dtype=np.int)
        if not np.array_equal(np.sort(self._labeling), np.arange(self._order)):
            raise ValueError("The labeling must be a permutation of [0 : order)")
        self._mapping = {symbol: tuple(int2binlist(label, width=self._bits_per_symbol)) \
                         for (symbol, label) in enumerate(self._labeling)}
        self._inverse_mapping = dict((value, key) for key, value in self._mapping.items())

        self._channel_snr = 1.0
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

    def __repr__(self):
        args = 'constellation={}, labeling={}'.format(self._constellation.tolist(), self._labeling.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

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
        The number :math:`m` of bits per symbol of the modulation. It is given by :math:`m = \log_2 M`, where :math:`M` is the order of the modulation. This property is read-only.
        """
        return self._bits_per_symbol

    @property
    def energy_per_symbol(self):
        """
        The average symbol energy :math:`E_\\mathrm{s}` of the constellation. It assumes equiprobable symbols. It is given by

        .. math::

            E_\\mathrm{s} = \\frac{1}{M} \\sum_{s_i \\in \\mathcal{S}} |s_i|^2,

        where :math:`|s_i|^2` is the energy of symbol :math:`s_i \\in \\mathcal{S}` and :math:`M` is the order of the modulation. This property is read-only.
        """
        return np.real(np.dot(self._constellation, self._constellation.conj())) / self._order

    @property
    def energy_per_bit(self):
        """
        The average bit energy :math:`E_\\mathrm{b}` of the constellation. It assumes equiprobable symbols. It is given by :math:`E_\\mathrm{b} = E_\\mathrm{s} / m`, where :math:`E_\\mathrm{s}` is the average symbol energy, and :math:`m` is the number of bits per symbol of the modulation.
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
        The signal-to-noise ratio :math:`\\mathrm{SNR}` of the channel. This is used in soft-decision methods. This is a read-and-write property.
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
            The bits to be converted. It should be a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length must be a multiple of :math:`m`.

        **Output:**

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols corresponding to :code:`bits`. It is a 1D-array of integers in the set :math:`[0 : M)`. Its length is equal to the length of :code:`bits` divided by :math:`m`.
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

        **Input:**

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols to be converted. It should be a 1D-array of integers in the set :math:`[0 : M)`. It may be of any length.

        **Output:**

        :code:`bits` : 1D-array of :obj:`int`
            The bits corresponding to :code:`symbols`. It is a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length is equal to the length of :code:`symbols` multiplied by :math:`m`.
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

    def _hard_symbol_demodulator(self, received):
        """General minimum distance hard demodulator."""
        mpoints, mconst = np.meshgrid(received, self._constellation)
        return np.argmin(np.absolute(mpoints - mconst), axis=0)

    def _soft_bit_demodulator(self, received):
        """Computes L-values of received points"""
        m = self._bits_per_symbol

        def pdf_recv_given_bit(bit_index, bit_value):
            bits = np.empty(m, dtype=np.int)
            bits[bit_index] = bit_value
            rest_index = np.setdiff1d(np.arange(m), [bit_index])
            f = 0.0
            for b_rest in itertools.product([0, 1], repeat=m-1):
                bits[rest_index] = b_rest
                point = self._constellation[self._inverse_mapping[tuple(bits)]]
                f += np.exp(-np.abs(received - point)**2 / self._channel_N0)
            return f

        soft_bits = np.empty(len(recv)*m, dtype=np.float)
        for bit_index in range(m):
            p0 = pdf_recv_given_bit(bit_index, 0)
            p1 = pdf_recv_given_bit(bit_index, 1)
            soft_bits[bit_index::m] = np.log(p0 / p1)

        return soft_bits

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
            raise ValueError("Parameter 'decision_method' should be either 'hard' or 'soft'")

    @staticmethod
    def _labeling_natural(order):
        labeling = np.arange(order)
        return labeling

    @staticmethod
    def _labeling_reflected(order):
        labeling = np.arange(order)
        labeling ^= (labeling >> 1)
        return labeling

    @staticmethod
    def _labeling_reflected_2d(order_I, order_Q):
        labeling_I = _Modulation._labeling_reflected(order_I)
        labeling_Q = _Modulation._labeling_reflected(order_Q)
        labeling = np.empty(order_I * order_Q, dtype=np.int)
        for i, (i_Q, i_I) in enumerate(itertools.product(labeling_Q, labeling_I)):
            labeling[i] = i_I + order_I * i_Q
        return labeling


class RealModulation(_Modulation):
    """
    General real modulation scheme. A *real modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of real numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.

    .. rubric:: Examples

    >>> mod = komm.RealModulation(constellation=[-0.5, 0, 0.5, 2], labeling=[0, 1, 3, 2])
    >>> mod.constellation
    array([-0.5,  0. ,  0.5,  2. ])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([-0.5,  0.5, -0.5,  0. ,  0. ])
    """
    def __init__(self, constellation, labeling):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`complex`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array containing :math:`M` real numbers.

        :code:`labeling` : 1D-array of :obj:`int`
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Must be a 1D-array of integers corresponding to a permutation of :math:`[0 : M)`.
        """
        super().__init__(np.array(constellation, dtype=np.float), labeling)


class PAModulation(RealModulation):
    """
    Pulse-amplitude modulation (PAM). It is a real modulation scheme (:class:`RealModulation`) in which the points of the constellation :math:`\\mathcal{S}` are *uniformly arranged* in the real line. More precisely,

    .. math::
        \\mathcal{S} = \\{ \\pm (2i + 1)A : i \\in [0 : M) \\},

    where :math:`M` is the *order* (a power of :math:`2`), and :math:`A` is the *base amplitude*.

    .. rubric:: Examples

    >>> pam = komm.PAModulation(4, base_amplitude=2.0)
    >>> pam.constellation
    array([-6., -2.,  2.,  6.])
    >>> pam.labeling
    array([0, 1, 3, 2])
    >>> pam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([-6.,  2., -6., -2., -2.])
    """
    def __init__(self, order, base_amplitude=1.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).
        """
        constellation = base_amplitude * np.arange(-order + 1, order, step=2, dtype=np.int)

        if isinstance(labeling, str):
            if labeling in ['natural', 'reflected']:
                labeling = getattr(_Modulation, '_labeling_' + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._base_amplitude = float(base_amplitude)

    def __repr__(self):
        args = '{}, base_amplitude={}'.format(self._order, self._base_amplitude)
        return '{}({})'.format(self.__class__.__name__, args)


class ComplexModulation(_Modulation):
    """
    General complex modulation scheme. A *complex modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of complex numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.

    .. rubric:: Examples

    >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
    >>> mod.constellation
    array([ 0.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
    >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j, -1.+0.j])
    """
    def __init__(self, constellation, labeling):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`complex`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array containing :math:`M` complex numbers.

        :code:`labeling` : 1D-array of :obj:`int`
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Must be a 1D-array of integers corresponding to a permutation of :math:`[0 : M)`.
        """
        super().__init__(np.array(constellation, dtype=np.complex), labeling)


class ASKModulation(ComplexModulation):
    """
    Amplitude-shift keying (ASK) modulation. It is a complex modulation scheme (:class:`ComplexModulation`) in which the points of the constellation :math:`\\mathcal{S}` are *uniformly arranged* in a ray. More precisely,

    .. math::

        \\mathcal{S} = \\{ iA \\exp(\\mathrm{j}\\phi): i \\in [0 : M) \\},

    where :math:`M` is the *order* (a power of :math:`2`), :math:`A` is the *base amplitude*, and :math:`\\phi` is the *phase offset* of the modulation.

    .. rubric:: Examples

    >>> ask = komm.ASKModulation(4, base_amplitude=2.0)
    >>> ask.constellation
    array([0.+0.j, 2.+0.j, 4.+0.j, 6.+0.j])
    >>> ask.labeling
    array([0, 1, 3, 2])
    >>> ask.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([0.+0.j, 4.+0.j, 0.+0.j, 2.+0.j, 2.+0.j])
    >>> ask.demodulate([(0.99+0.3j), (1.01-0.5j), (4.99+0.7j), (5.01-0.9j)])
    array([0, 0, 1, 0, 1, 1, 0, 1])
    """
    def __init__(self, order, base_amplitude=1.0, phase_offset=0.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).
        """
        constellation = base_amplitude * np.arange(order, dtype=np.int) * np.exp(1j*phase_offset)

        if isinstance(labeling, str):
            if labeling in ['natural', 'reflected']:
                labeling = getattr(_Modulation, '_labeling_' + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._base_amplitude = float(base_amplitude)
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        args = '{}, base_amplitude={}, phase_offset={}'.format(self._order, self._base_amplitude, self._phase_offset)
        return '{}({})'.format(self.__class__.__name__, args)


class PSKModulation(ComplexModulation):
    """
    Phase-shift keying (PSK) modulation. It is a complex modulation scheme (:class:`ComplexModulation`) in which the points of the constellation :math:`\\mathcal{S}` are *uniformly arranged* in a circle. More precisely,

    .. math::
        \\mathcal{S} = \\left \\{ A \\exp \\left( \mathrm{j} \\frac{2 \\pi i}{M} \\right) \\exp(\\mathrm{j} \\phi) : i \\in [0 : M) \\right \\}

    where :math:`M` is the *order* (a power of :math:`2`), :math:`A` is the *amplitude*, and :math:`\\phi` is the *phase offset* of the modulation.

    .. rubric:: Examples

    >>> psk = komm.PSKModulation(4, phase_offset=np.pi/4)
    >>> psk.constellation
    array([ 0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678-0.70710678j])
    >>> psk.labeling
    array([0, 1, 3, 2])
    >>> psk.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    array([ 0.70710678+0.70710678j, -0.70710678-0.70710678j, 0.70710678+0.70710678j, -0.70710678+0.70710678j, 0.70710678+0.70710678j])
    """
    def __init__(self, order, amplitude=1.0, phase_offset=0.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`amplitude` : :obj:`float`, optional
            The amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).
        """
        constellation = np.exp(2j*np.pi*np.arange(order) / order) * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling in ['natural', 'reflected']:
                labeling = getattr(_Modulation, '_labeling_' + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._amplitude = float(amplitude)
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        args = '{}, amplitude={}, phase_offset={}'.format(self._order, self._amplitude, self._phase_offset)
        return '{}({})'.format(self.__class__.__name__, args)


class APSKModulation(ComplexModulation):
    """
    Amplitude- and phase-shift keying (APSK) modulation.
    """
    def __init__(self):
       """
       Here.
       """
       pass


class QAModulation(ComplexModulation):
    """
    Quadratude-amplitude modulation (QAM). It is a complex modulation scheme (:class:`ComplexModulation`) in which the constellation :math:`\\mathcal{S}` is given as a cartesian product of two PAM (:class:`PAModulation`) constellations, namely, the *in-phase constellation*, and the *quadrature constellation*. More precisely,

    .. math::
        \\mathcal{S} = \\{ [\\pm(2i_\\mathrm{I} + 1)A_\\mathrm{I} \\pm \\mathrm{j}(2i_\\mathrm{Q} + 1)A_\\mathrm{Q}] \\exp(\\mathrm{j}\\phi) : i_\\mathrm{I} \\in [0 : M_\\mathrm{I}), i_\\mathrm{Q} \\in [0 : M_\\mathrm{Q}) \\},

    where :math:`M_\\mathrm{I}` and :math:`M_\\mathrm{Q}` are the *orders* (powers of :math:`2`), and :math:`A_\\mathrm{I}` and :math:`A_\\mathrm{Q}` are the *base amplitudes* of the in-phase and quadratude constellations, respectively. Also, :math:`\\phi` is the *phase offset*. The size of the resulting complex-valued constellation is :math:`M = M_\\mathrm{I} M_\\mathrm{Q}`, a power of :math:`2`.

    .. image:: figures/qam_16.png
       :alt: 16-QAM constellation
       :align: center

    .. rubric:: Examples

    >>> qam = komm.QAModulation(16)
    >>> qam.constellation
    array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j, -3.-1.j, -1.-1.j,  1.-1.j, 3.-1.j, -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j, -3.+3.j, -1.+3.j, 1.+3.j,  3.+3.j])
    >>> qam.labeling
    array([ 0,  1,  3,  2,  4,  5,  7,  6, 12, 13, 15, 14,  8,  9, 11, 10])
    >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0])
    array([-3.+1.j, -3.-1.j])

    >>> qam = komm.QAModulation(orders=(4, 2), base_amplitudes=(1.0, 3.0))
    >>> qam.constellation
    array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j, -3.+3.j, -1.+3.j,  1.+3.j, 3.+3.j])
    >>> qam.labeling
    array([0, 1, 3, 2, 4, 5, 7, 6])
    >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1])
    array([-3.+3.j, -1.-3.j, -1.+3.j])
    """
    def __init__(self, orders, base_amplitudes=1.0, phase_offset=0.0, labeling='reflected_2d'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`orders` : :obj:`(int, int)` or :obj:`int`
            A tuple :math:`(M_\\mathrm{I}, M_\\mathrm{Q})` with the orders of the in-phase and quadrature constellations, respectively; both :math:`M_\\mathrm{I}` and :math:`M_\\mathrm{Q}` must be powers of :math:`2`. If specified as a single integer :math:`M`, then it is assumed that :math:`M_\\mathrm{I} = M_\\mathrm{Q} = \\sqrt{M}`; in this case, :math:`M` must be an square power of :math:`2`.

        :code:`base_amplitudes` : :obj:`(float, float)` or :obj:`float`, optional
            A tuple :math:`(A_\\mathrm{I}, A_\\mathrm{Q})` with the base amplitudes of the in-phase and quadrature constellations, respectively.  If specified as a single float :math:`A`, then it is assumed that :math:`A_\\mathrm{I} = A_\\mathrm{Q} = A`. The default value is :math:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected_2d'`. The default value is :code:`'reflected_2d'` (Gray code).
        """
        if isinstance(orders, (tuple, list)):
            order_I, order_Q = int(orders[0]), int(orders[1])
            self._orders = (order_I, order_Q)
        else:
            order_I = order_Q = int(np.sqrt(orders))
            self._orders = int(orders)

        if isinstance(base_amplitudes, (tuple, list)):
            base_amplitude_I, base_amplitude_Q = float(base_amplitudes[0]), float(base_amplitudes[1])
            self._base_amplitudes = (base_amplitude_I, base_amplitude_Q)
        else:
            base_amplitude_I = base_amplitude_Q = float(base_amplitudes)
            self._base_amplitudes = base_amplitude_I

        constellation_I = base_amplitude_I * np.arange(-order_I + 1, order_I, step=2, dtype=np.int)
        constellation_Q = base_amplitude_Q * np.arange(-order_Q + 1, order_Q, step=2, dtype=np.int)
        constellation = (constellation_I + 1j*constellation_Q[np.newaxis].T).flatten() * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling in ['natural', 'reflected_2d']:
                labeling = getattr(_Modulation, '_labeling_' + labeling)(order_I, order_Q)
            else:
                raise ValueError("Only 'natural' or 'reflected_2d' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._orders = orders
        self._base_amplitudes = base_amplitudes
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        args = '{}, base_amplitudes={}, phase_offset={}'.format(self._orders, self._base_amplitudes, self._phase_offset)
        return '{}({})'.format(self.__class__.__name__, args)



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
