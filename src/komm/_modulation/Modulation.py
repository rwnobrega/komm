from itertools import combinations
from typing import final

import numpy as np
import numpy.typing as npt

from .._util.decorators import vectorize


class Modulation:
    r"""
    General modulation scheme. A *modulation scheme* of *order* $M = 2^m$ is defined by a *constellation* $\mathbf{X}$, which is a real or complex vector of length $M$, and a *binary labeling* $\mathbf{Q}$, which is an $M \times m$ binary matrix whose rows are all distinct. The $i$-th element of $\mathbf{X}$, for $i \in [0:M)$, is denoted by $x_i$ and is called the $i$-th *constellation symbol*. The $i$-th row of $\mathbf{Q}$, for $i \in [0:M)$, is called the *binary representation* of the $i$-th constellation symbol. For more details, see <cite>SA15, Sec. 2.5</cite>.

    Parameters:
        constellation: The constellation $\mathbf{X}$ of the modulation. Must be a 1D-array containing $M$ real or complex numbers.

        labeling: The binary labeling $\mathbf{Q}$ of the modulation. Must be a 2D-array of shape $(M, m)$ where each row is a distinct binary $m$-tuple.

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

    def __init__(self, constellation: npt.ArrayLike, labeling: npt.ArrayLike) -> None:
        self.constellation = np.asarray(constellation)
        self.labeling = np.asarray(labeling)
        order, m = self.order, self.bits_per_symbol
        if order & (order - 1):
            raise ValueError("length of 'constellation' must be a power of two")
        if self.labeling.shape != (order, m):
            raise ValueError(f"shape of 'labeling' must be ({order}, {m})")
        if np.any(self.labeling < 0) or np.any(self.labeling > 1):
            raise ValueError("elements of 'labeling' must be either 0 or 1")
        if len(set(tuple(row) for row in self.labeling)) != order:
            raise ValueError("rows of 'labeling' must be distinct")

    def __repr__(self) -> str:
        args = ", ".join([
            f"constellation={self.constellation.tolist()}",
            f"labeling={self.labeling.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

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

    @final
    def modulate(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        r"""
        Modulates one or more sequences of bits to their corresponding constellation symbols (real or complex numbers).

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $m$, or a multidimensional array where the last dimension is a multiple of $m$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension divided by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.modulate([0, 0, 1, 1, 0, 0, 1, 0])
            array([ 2. ,  0. ,  2. , -0.5])
            >>> modulation.modulate([[0, 0, 1, 1], [0, 0, 1, 0]])
            array([[ 2. ,  0. ],
                   [ 2. , -0.5]])
        """
        input = np.asarray(input)
        if input.shape[-1] % self.bits_per_symbol != 0:
            raise ValueError(
                "last dimension of 'bits' must be a multiple of bits per symbol"
                f" {self.bits_per_symbol} (got {input.shape[-1]})"
            )
        bits = input.reshape(*input.shape[:-1], -1, self.bits_per_symbol)
        symbols = vectorize(self._modulate)(bits)
        output = symbols.reshape(*symbols.shape[:-1], -1)
        return output

    def _modulate(
        self, bits: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        return self.constellation[self.inverse_labeling[tuple(bits)]]

    @final
    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Demodulates one or more sequences of received points (real or complex numbers) to their corresponding sequences of hard bits ($\mathtt{0}$ or $\mathtt{1}$) using hard-decision decoding.

        Parameters:
            input: The input sequence(s). Can be either a single sequence, or a multidimensional array.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension multiplied by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.demodulate_hard([2.17, -0.06, 1.94, -0.61])
            array([0, 0, 1, 1, 0, 0, 1, 0])
            >>> modulation.demodulate_hard([[2.17, -0.06], [1.94, -0.61]])
            array([[0, 0, 1, 1],
                   [0, 0, 1, 0]])
        """
        input = np.asarray(input)
        received = input.reshape(*input.shape[:-1], -1, 1)
        hard_bits = vectorize(self._demodulate_hard)(received)
        output = hard_bits.reshape(*input.shape[:-1], -1)
        return output

    def _demodulate_hard(
        self, received: npt.NDArray[np.floating | np.complexfloating]
    ) -> npt.NDArray[np.integer]:
        # General minimum Euclidean distance hard demodulator.
        return self.labeling[np.argmin(np.abs(self.constellation - received[0]))]

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        r"""
        Demodulates one or more sequences of received points (real or complex numbers) to their corresponding sequences of soft bits (L-values) using soft-decision decoding. The soft bits are the log-likelihood ratios of the bits, where positive values correspond to bit $\mathtt{0}$ and negative values correspond to bit $\mathtt{1}$.

        Parameters:
            input: The received sequence(s). Can be either a single sequence, or a multidimensional array.
            snr: The signal-to-noise ratio (SNR) of the channel. It should be a positive real number. The default value is `1.0`.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension multiplied by $m$.

        Examples:
            >>> modulation = komm.Modulation(constellation=[-0.5, 0.0, 0.5, 2.0], labeling=[[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> modulation.demodulate_soft([2.17, -0.06, 1.94, -0.61], snr=100.0).round(1)
            array([ 416. ,  245.3,  -27.6,  -16.9,  334.2,  184. , -108.4,   32. ])
            >>> modulation.demodulate_soft([[2.17, -0.06], [1.94, -0.61]], snr=100.0).round(1)
            array([[ 416. ,  245.3,  -27.6,  -16.9],
                   [ 334.2,  184. , -108.4,   32. ]])
        """
        input = np.asarray(input)
        received = input.reshape(*input.shape[:-1], -1, 1)
        soft_bits = self._demodulate_soft(received, snr)
        output = soft_bits.reshape(*input.shape[:-1], -1)
        return output

    def _demodulate_soft(
        self, received: npt.NDArray[np.floating | np.complexfloating], snr: float
    ) -> npt.NDArray[np.floating]:
        # Computes the L-values (LLR) of each bit. Assumes uniformly distributed bits.
        # See [SA15, eq. (3.50)].
        m = self.bits_per_symbol
        n0 = self.energy_per_symbol / snr
        received = received.ravel()

        # Precompute the distances and exponentials.
        distances = np.abs(received[:, np.newaxis] - self.constellation) ** 2
        exp_terms = np.exp(-distances / n0)

        soft_bits = np.empty(received.size * m, dtype=float)
        for bit_index in range(m):
            i0 = [i for i, label in enumerate(self.labeling) if label[bit_index] == 0]
            i1 = [i for i, label in enumerate(self.labeling) if label[bit_index] == 1]
            p0 = np.sum(exp_terms[:, i0], axis=1)
            p1 = np.sum(exp_terms[:, i1], axis=1)
            with np.errstate(divide="ignore"):
                soft_bits[bit_index::m] = np.log(p0) - np.log(p1)
        return soft_bits
