import itertools

import numpy as np

from .._util import int2binlist


class Modulation:
    def __init__(self, constellation, labeling):
        self._constellation = np.array(constellation)
        self._order = self._constellation.size
        if self._order & (self._order - 1):
            raise ValueError("The length of constellation must be a power of two")
        self._bits_per_symbol = (self._order - 1).bit_length()

        self._labeling = np.array(labeling, dtype=int)
        if not np.array_equal(np.sort(self._labeling), np.arange(self._order)):
            raise ValueError("The labeling must be a permutation of [0 : order)")
        self._mapping = {
            symbol: tuple(int2binlist(label, width=self._bits_per_symbol))
            for (symbol, label) in enumerate(self._labeling)
        }
        self._inverse_mapping = dict((value, key) for key, value in self._mapping.items())

        self._channel_snr = 1.0
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

    def __repr__(self):
        args = "constellation={}, labeling={}".format(self._constellation.tolist(), self._labeling.tolist())
        return "{}({})".format(self.__class__.__name__, args)

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
        The number :math:`m` of bits per symbol of the modulation. It is given by :math:`m = \\log_2 M`, where :math:`M` is the order of the modulation. This property is read-only.
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
        Converts bits to symbols using the modulation binary labeling.

        .. rubric:: Input

        :code:`bits` : 1D-array of :obj:`int`
            The bits to be converted. It should be a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length must be a multiple of :math:`m`.

        .. rubric:: Output

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols corresponding to :code:`bits`. It is a 1D-array of integers in the set :math:`[0 : M)`. Its length is equal to the length of :code:`bits` divided by :math:`m`.
        """
        m = self._bits_per_symbol
        n_symbols = len(bits) // m
        assert len(bits) == n_symbols * m
        symbols = np.empty(n_symbols, dtype=int)
        for i, bit_sequence in enumerate(np.reshape(bits, newshape=(n_symbols, m))):
            symbols[i] = self._inverse_mapping[tuple(bit_sequence)]
        return symbols

    def symbols_to_bits(self, symbols):
        """
        Converts symbols to bits using the modulation binary labeling.

        .. rubric:: Input

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols to be converted. It should be a 1D-array of integers in the set :math:`[0 : M)`. It may be of any length.

        .. rubric:: Output

        :code:`bits` : 1D-array of :obj:`int`
            The bits corresponding to :code:`symbols`. It is a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length is equal to the length of :code:`symbols` multiplied by :math:`m`.
        """
        m = self._bits_per_symbol
        n_bits = len(symbols) * m
        bits = np.empty(n_bits, dtype=int)
        for i, symbol in enumerate(symbols):
            bits[i * m : (i + 1) * m] = self._mapping[symbol]
        return bits

    def modulate(self, bits):
        """
        Modulates a sequence of bits to its corresponding constellation points.
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

        def pdf_received_given_bit(bit_index, bit_value):
            bits = np.empty(m, dtype=int)
            bits[bit_index] = bit_value
            rest_index = np.setdiff1d(np.arange(m), [bit_index])
            f = 0.0
            for b_rest in itertools.product([0, 1], repeat=m - 1):
                bits[rest_index] = b_rest
                point = self._constellation[self._inverse_mapping[tuple(bits)]]
                f += np.exp(-np.abs(received - point) ** 2 / self._channel_N0)
            return f

        soft_bits = np.empty(len(received) * m, dtype=float)
        for bit_index in range(m):
            p0 = pdf_received_given_bit(bit_index, 0)
            p1 = pdf_received_given_bit(bit_index, 1)
            soft_bits[bit_index::m] = np.log(p0 / p1)

        return soft_bits

    def demodulate(self, received, decision_method="hard"):
        """
        Demodulates a sequence of received points to a sequence of bits.
        """
        if decision_method == "hard":
            symbols_hat = self._hard_symbol_demodulator(received)
            return self.symbols_to_bits(symbols_hat)
        elif decision_method == "soft":
            return self._soft_bit_demodulator(received)
        else:
            raise ValueError("Parameter 'decision_method' should be either 'hard' or 'soft'")

    @staticmethod
    def _labeling_natural(order):
        labeling = np.arange(order)
        return labeling

    @staticmethod
    def _labeling_reflected(order):
        labeling = np.arange(order)
        labeling ^= labeling >> 1
        return labeling

    @staticmethod
    def _labeling_reflected_2d(order_I, order_Q):
        labeling_I = Modulation._labeling_reflected(order_I)
        labeling_Q = Modulation._labeling_reflected(order_Q)
        labeling = np.empty(order_I * order_Q, dtype=int)
        for i, (i_Q, i_I) in enumerate(itertools.product(labeling_Q, labeling_I)):
            labeling[i] = i_I + order_I * i_Q
        return labeling
