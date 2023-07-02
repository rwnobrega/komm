import itertools as it

import numpy as np

from .._util import cartesian_product, int2binlist


class Modulation:
    r"""
    General modulation scheme. A *modulation scheme* of *order* $M = 2^m$ is defined by a *constellation* $\mathbf{X}$, which is a real or complex vector of length $M$, and a *binary labeling* $\mathbf{Q}$, which is an $M \times m$ binary matrix whose rows are all distinct. The $i$-th element of $\mathbf{X}$, for $i \in [0:M)$, is denoted by $x_i$ and is called the $i$-th *constellation symbol*. The $i$-th row of $\mathbf{Q}$, for $i \in [0:M)$, is called the *binary representation* of the $i$-th constellation symbol.

    For more details, see <cite>SA15, Sec. 2.5</cite>.
    """

    def __init__(self, constellation, labeling):
        r"""
        Constructor for the class.

        Parameters:

            constellation (Array1D[float] | Array1D[complex]): The constellation $\mathbf{X}$ of the modulation. Must be a 1D-array containing $M$ real or complex numbers.

            labeling (Array2D[int]): The binary labeling $\mathbf{Q}$ of the modulation. Must be a 2D-array of shape $(M, m)$ where each row is a distinct binary $m$-tuple.

        Examples:

            The real modulation scheme depicted in the figure below has $M = 4$ and $m = 2$.

            <figure markdown>
              ![Example for real modulation with M = 4](/figures/modulation_real_4.svg)
            </figure>

            The constellation is given by
            $$
                \mathbf{X} = \begin{bmatrix}
                    -0.5 \\\\
                    0.0 \\\\
                    0.5 \\\\
                    2.0
                \end{bmatrix},
            $$
            and the binary labeling is given by
            $$
                \mathbf{Q} = \begin{bmatrix}
                    1 & 0 \\\\
                    1 & 1 \\\\
                    0 & 1 \\\\
                    0 & 0
                \end{bmatrix}.
            $$

            >>> komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])

            The complex modulation scheme depicted in the figure below has $M = 4$ and $m = 2$.

            <figure markdown>
              ![Example for complex modulation with M = 4](/figures/modulation_complex_4.svg)
            </figure>

            The constellation is given by
            $$
                \mathbf{X} = \begin{bmatrix}
                    0  \\\\
                    -1  \\\\
                    1  \\\\
                    \mathrm{j}
                \end{bmatrix},
            $$
            and the binary labeling is given by
            $$
                \mathbf{Q} = \begin{bmatrix}
                    0 & 0 \\\\
                    0 & 1 \\\\
                    1 & 0 \\\\
                    1 & 1
                \end{bmatrix}.
            $$

            >>> komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            Modulation(constellation=[0j, (-1+0j), (1+0j), 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
        """
        if np.isrealobj(constellation):
            self._constellation = np.array(constellation, dtype=float)
        else:
            self._constellation = np.array(constellation, dtype=complex)
        self._order = self._constellation.size
        if self._order & (self._order - 1):
            raise ValueError("The length of `constellation` must be a power of two")
        self._bits_per_symbol = (self._order - 1).bit_length()

        self._labeling = np.array(labeling, dtype=int)
        if self._labeling.shape != (self._order, self._bits_per_symbol):
            raise ValueError("The shape of `labeling` must be ({}, {})".format(self._order, self._bits_per_symbol))
        if np.any(self._labeling < 0) or np.any(self._labeling > 1):
            raise ValueError("The elements of `labeling` must be either 0 or 1")
        if len(set(tuple(row) for row in self._labeling)) != self._order:
            raise ValueError("The rows of `labeling` must be distinct")

        self._inverse_labeling = dict(zip(map(tuple, self._labeling), range(self._order)))

    def __repr__(self):
        args = "constellation={}, labeling={}".format(self._constellation.tolist(), self._labeling.tolist())
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def constellation(self):
        r"""
        The constellation $\mathbf{X}$ of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.constellation
            array([-0.5,  0. ,  0.5,  2. ])
        """
        return self._constellation

    @property
    def labeling(self):
        r"""
        The binary labeling $\mathbf{Q}$ of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.labeling
            array([[1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]])
        """
        return self._labeling

    @property
    def order(self):
        r"""
        The order $M$ of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.order
            4
        """
        return self._order

    @property
    def bits_per_symbol(self):
        r"""
        The number $m$ of bits per symbol of the modulation. It is given by $m = \log_2 M$, where $M$ is the order of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.bits_per_symbol
            2
        """
        return self._bits_per_symbol

    @property
    def energy_per_symbol(self):
        r"""
        The average symbol energy $E_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
            E_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} |x_i|^2,
        $$
        where $|x_i|^2$ is the energy of constellation symbol $x_i$, and $M$ is the order of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.energy_per_symbol
            1.125

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.energy_per_symbol
            0.75
        """
        return np.real(np.dot(self._constellation, self._constellation.conj())) / self._order

    @property
    def energy_per_bit(self):
        r"""
        The average bit energy $E_\mathrm{b}$ of the constellation. It assumes equiprobable symbols. It is given by $E_\mathrm{b} = E_\mathrm{s} / m$, where $E_\mathrm{s}$ is the average symbol energy, and $m$ is the number of bits per symbol of the modulation.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.energy_per_bit
            0.5625

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.energy_per_bit
            0.375
        """
        return self.energy_per_symbol / np.log2(self._order)

    @property
    def symbol_mean(self):
        r"""
        The mean $\mu_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
            \mu_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} x_i.
        $$

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.symbol_mean
            0.5

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.symbol_mean
            0.25j
        """
        return np.sum(self._constellation) / self._order

    @property
    def minimum_distance(self):
        r"""
        The minimum Euclidean distance $d_\mathrm{min}$ of the constellation. It is given by
        $$
            d_\mathrm{min} = \min_{i, j \in [0:M), ~ i \neq j} |x_i - x_j|.
        $$

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.minimum_distance
            0.5

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.minimum_distance
            1.0
        """
        return np.min(
            np.fromiter(
                (np.abs(s1 - s2) for s1, s2 in it.combinations(self._constellation, 2)),
                dtype=float,
            )
        )

    def modulate(self, bits):
        r"""
        Modulates a sequence of bits to its corresponding constellation symbols.

        Parameters:

            bits (Array1D[int]): The bits to be modulated. It should be a 1D-array of integers in the set $\\{ 0, 1 \\}$. Its length must be a multiple of $m$.

        Returns:

            symbols (Array1D[complex] | Array1D[float]): The constellation symbols corresponding to `bits`. It is a 1D-array of real or complex numbers. Its length is equal to the length of `bits` divided by $m$.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0])
            array([ 2. ,  0. ,  2. , -0.5])
        """
        m = self._bits_per_symbol
        n_symbols = len(bits) // m
        if len(bits) != n_symbols * m:
            raise ValueError("The length of `bits` must be a multiple of the number of bits per symbol.")
        symbols = np.empty(n_symbols, dtype=self._constellation.dtype)
        for i, bit_sequence in enumerate(np.reshape(bits, newshape=(n_symbols, m))):
            symbols[i] = self._constellation[self._inverse_labeling[tuple(bit_sequence)]]
        return symbols

    def _demodulate_hard(self, received):
        # General minimum Euclidean distance hard demodulator.
        hard_bits = np.empty((len(received), self._bits_per_symbol), dtype=int)
        for i, y in enumerate(received):
            hard_bits[i, :] = self._labeling[np.argmin(np.abs(self._constellation - y)), :]
        return np.reshape(hard_bits, newshape=-1)

    def _demodulate_soft(self, received, channel_snr=1.0):
        # Computes the L-values (LLR) of each bit.
        # Assumes uniformly distributed bits. See SA15, eq. (3.50).
        m = self._bits_per_symbol
        N0 = self.energy_per_symbol / channel_snr

        def pdf_received_given_bit(bit_index, bit_value):
            bits = np.empty(m, dtype=int)
            bits[bit_index] = bit_value
            rest_index = np.setdiff1d(np.arange(m), [bit_index])
            f = 0.0
            for b_rest in it.product([0, 1], repeat=m - 1):
                bits[rest_index] = b_rest
                point = self._constellation[self._inverse_labeling[tuple(bits)]]
                f += np.exp(-np.abs(received - point) ** 2 / N0)
            return f

        soft_bits = np.empty(len(received) * m, dtype=float)
        for bit_index in range(m):
            p0 = pdf_received_given_bit(bit_index, 0)
            p1 = pdf_received_given_bit(bit_index, 1)
            soft_bits[bit_index::m] = np.log(p0 / p1)

        return soft_bits

    def demodulate(self, received, decision_method="hard", **kwargs):
        r"""
        Demodulates a sequence of received points to a sequence of bits.

        Parameters:

            received (Array1D[complex] | Array1D[float]): The received points to be demodulated. It should be a 1D-array of real or complex numbers. It may be of any length.

            decision_method (str): The decision method to be used. It should be either `'hard'` (corresponding to *hard-decision decoding*) or `'soft'` (corresponding to *soft-decision decoding*). The default value is `'hard'`.

            kwargs (): Keyword arguments to be passed to the demodulator.

        Returns:

            bits_or_soft_bits (Array1D[int] | Array1D[float]): The (hard or soft) bits corresponding to `received`. In the case of hard-decision decoding, it is a 1D-array of bits (integers in the set $\\{ 0, 1 \\}$); in the case of of soft-decision decoding, it is a 1D-array of L-values (real numbers, where positive values correspond to bit $0$ and negative values correspond to bit $1$). Its length is equal to the length of `received` multiplied by $m$.

        Examples:

            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> received = [2.17, -0.06, 1.94, -0.61]

            >>> mod.demodulate(received)
            array([0, 0, 1, 1, 0, 0, 1, 0])

            >>> mod.demodulate(received, decision_method='soft', channel_snr=100.0)
            array([ 416.        ,  245.33333333,  -27.5555556 ,  -16.88888889,
                    334.22222222,  184.        , -108.44444444,   32.        ])

        """
        if decision_method in ["hard", "soft"]:
            demodulate = getattr(self, "_demodulate_" + decision_method)
        else:
            raise ValueError("Parameter `decision_method` should be either 'hard' or 'soft'")
        return demodulate(np.asarray(received), **kwargs)
