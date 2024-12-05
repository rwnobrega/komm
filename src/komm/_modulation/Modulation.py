from itertools import combinations, product

import numpy as np
import numpy.typing as npt
from attrs import define, field


@define
class Modulation:
    r"""
    General modulation scheme. A *modulation scheme* of *order* $M = 2^m$ is defined by a *constellation* $\mathbf{X}$, which is a real or complex vector of length $M$, and a *binary labeling* $\mathbf{Q}$, which is an $M \times m$ binary matrix whose rows are all distinct. The $i$-th element of $\mathbf{X}$, for $i \in [0:M)$, is denoted by $x_i$ and is called the $i$-th *constellation symbol*. The $i$-th row of $\mathbf{Q}$, for $i \in [0:M)$, is called the *binary representation* of the $i$-th constellation symbol. For more details, see <cite>SA15, Sec. 2.5</cite>.

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

        >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
        >>> modulation.constellation
        array([-0.5,  0. ,  0.5,  2. ])
        >>> modulation.labeling
        array([[1, 0],
               [1, 1],
               [0, 1],
               [0, 0]])

        The complex modulation scheme depicted in the figure below has $M = 4$ and $m = 2$.

        <figure markdown>
          ![Example for complex modulation with M = 4](/figures/modulation_complex_4.svg)
        </figure>

        The constellation is given by
        $$
          \mathbf{X} = \begin{bmatrix}
             0 \\\\
            -1 \\\\
             1 \\\\
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

        >>> modulation = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> modulation.constellation
        array([ 0.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
        >>> modulation.labeling
        array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])
    """

    constellation: npt.NDArray[np.float64 | np.complex128] = field(
        converter=np.asarray,
        repr=lambda x: x.tolist(),
    )
    labeling: npt.NDArray[np.int_] = field(
        converter=np.asarray,
        repr=lambda x: x.tolist(),
    )

    def __attrs_post_init__(self) -> None:
        order, m = self.order, self.bits_per_symbol
        if order & (order - 1):
            raise ValueError("length of 'constellation' must be a power of two")
        if self.labeling.shape != (order, m):
            raise ValueError(f"shape of 'labeling' must be ({order}, {m})")
        if np.any(self.labeling < 0) or np.any(self.labeling > 1):
            raise ValueError("elements of 'labeling' must be either 0 or 1")
        if len(set(tuple(row) for row in self.labeling)) != order:
            raise ValueError("rows of 'labeling' must be distinct")

    @property
    def order(self) -> int:
        r"""
        The order $M$ of the modulation.

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.order
            4
        """
        return self.constellation.size

    @property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        The inverse labeling of the modulation. It is a dictionary that maps each binary tuple to the corresponding constellation index.

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.inverse_labeling
            {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
        """
        return {tuple(int(x) for x in row): i for i, row in enumerate(self.labeling)}

    @property
    def bits_per_symbol(self) -> int:
        r"""
        The number $m$ of bits per symbol of the modulation. It is given by $m = \log_2 M$, where $M$ is the order of the modulation.

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.bits_per_symbol
            2
        """
        return (self.order - 1).bit_length()

    @property
    def energy_per_symbol(self) -> float:
        r"""
        The average symbol energy $E_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
          E_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} |x_i|^2,
        $$
        where $|x_i|^2$ is the energy of constellation symbol $x_i$, and $M$ is the order of the modulation.

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.energy_per_symbol
            np.float64(1.125)

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.energy_per_symbol
            np.float64(0.75)
        """
        return ((self.constellation * self.constellation.conj()).real).mean()

    @property
    def energy_per_bit(self) -> float:
        r"""
        The average bit energy $E_\mathrm{b}$ of the constellation. It assumes equiprobable symbols. It is given by $E_\mathrm{b} = E_\mathrm{s} / m$, where $E_\mathrm{s}$ is the average symbol energy, and $m$ is the number of bits per symbol of the modulation.

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.energy_per_bit
            np.float64(0.5625)

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.energy_per_bit
            np.float64(0.375)
        """
        return self.energy_per_symbol / self.bits_per_symbol

    @property
    def symbol_mean(self) -> float | complex:
        r"""
        The mean $\mu_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
          \mu_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} x_i.
        $$

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.symbol_mean
            np.float64(0.5)

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.symbol_mean
            np.complex128(0.25j)
        """
        return self.constellation.mean()

    @property
    def minimum_distance(self) -> float:
        r"""
        The minimum Euclidean distance $d_\mathrm{min}$ of the constellation. It is given by
        $$
            d_\mathrm{min} = \min_{i, j \in [0:M), ~ i \neq j} |x_i - x_j|.
        $$

        Examples:
            >>> mod = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> mod.minimum_distance
            np.float64(0.5)

            >>> mod = komm.Modulation(constellation=[0, -1, 1, 1j], labeling=[[0, 0], [0, 1], [1, 0], [1, 1]])
            >>> mod.minimum_distance
            np.float64(1.0)
        """
        return np.min(
            [np.abs(s1 - s2) for s1, s2 in combinations(self.constellation, 2)]
        )

    def modulate(self, bits: npt.ArrayLike) -> npt.NDArray[np.float64 | np.complex128]:
        r"""
        Modulates a sequence of bits to its corresponding constellation symbols.

        Parameters:
            bits (Array1D[int]): The bits to be modulated. It should be a 1D-array of integers in the set $\{ 0, 1 \}$. Its length must be a multiple of $m$.

        Returns:
            symbols (Array1D[complex] | Array1D[float]): The constellation symbols corresponding to `bits`. It is a 1D-array of real or complex numbers. Its length is equal to the length of `bits` divided by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.modulate([0, 0, 1, 1, 0, 0, 1, 0])
            array([ 2. ,  0. ,  2. , -0.5])
        """
        bits = np.asarray(bits, dtype=int)
        m = self.bits_per_symbol
        n_symbols = bits.size // m
        if bits.size != n_symbols * m:
            raise ValueError("length of 'bits' must be a multiple of bits per symbol")
        symbols = np.empty(n_symbols, dtype=self.constellation.dtype)
        for i, bit_sequence in enumerate(np.reshape(bits, shape=(n_symbols, m))):
            symbols[i] = self.constellation[self.inverse_labeling[tuple(bit_sequence)]]
        return symbols

    def demodulate_hard(self, received: npt.ArrayLike) -> npt.NDArray[np.int_]:
        r"""
        Demodulates a sequence of received points to a sequence of bits using hard-decision decoding.

        Parameters:
            received (Array1D[T]): The received points to be demodulated. It should be a 1D-array of real or complex numbers. It may be of any length.

        Returns:
            hard_bits (Array1D[int]): The bits corresponding to `received`. It is a 1D-array of bits (integers in the set $\{ 0, 1 \}$). Its length is equal to the length of `received` multiplied by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.demodulate_hard([2.17, -0.06, 1.94, -0.61])
            array([0, 0, 1, 1, 0, 0, 1, 0])
        """
        # General minimum Euclidean distance hard demodulator.
        received = np.asarray(received)
        hard_bits = np.empty((received.size, self.bits_per_symbol), dtype=int)
        for i, y in enumerate(received):
            hard_bits[i, :] = self.labeling[
                np.argmin(np.abs(self.constellation - y)), :
            ]
        hard_bits = np.reshape(hard_bits, shape=-1)
        return hard_bits

    def demodulate_soft(
        self, received: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.float64]:
        r"""
        Demodulates a sequence of received points to a sequence of bits using soft-decision decoding.

        Parameters:
            received (Array1D[T]): The received points to be demodulated. It should be a 1D-array of real or complex numbers. It may be of any length.

            snr (float): The signal-to-noise ratio (SNR) of the channel. It should be a positive real number.

        Returns:
            soft_bits (Array1D[float]): The soft bits corresponding to `received`. It is a 1D-array of L-values (real numbers, where positive values correspond to bit $0$ and negative values correspond to bit $1$). Its length is equal to the length of `received` multiplied by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.demodulate_soft([2.17, -0.06, 1.94, -0.61], snr=100.0)
            array([ 416.        ,  245.33333333,  -27.5555556 ,  -16.88888889,
                    334.22222222,  184.        , -108.44444444,   32.        ])
        """
        # Computes the L-values (LLR) of each bit. Assumes uniformly distributed bits.
        # See [SA15, eq. (3.50)].
        m = self.bits_per_symbol
        n0 = self.energy_per_symbol / snr

        def pdf_received_given_bit(bit_index: int, bit_value: int) -> float:
            bits = np.empty(m, dtype=int)
            bits[bit_index] = bit_value
            rest_index = np.setdiff1d(np.arange(m), [bit_index])
            f = 0.0
            for b_rest in product([0, 1], repeat=m - 1):
                bits[rest_index] = b_rest
                point = self.constellation[self.inverse_labeling[tuple(bits)]]
                f += np.exp(-np.abs(received - point) ** 2 / n0)
            return f

        received = np.asarray(received)
        soft_bits = np.empty(received.size * m, dtype=float)
        for bit_index in range(m):
            p0 = pdf_received_given_bit(bit_index, 0)
            p1 = pdf_received_given_bit(bit_index, 1)
            with np.errstate(divide="ignore"):
                soft_bits[bit_index::m] = np.log(p0) - np.log(p1)
        return soft_bits
