from abc import ABC, abstractmethod
from functools import cached_property
from itertools import combinations
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import final

from .._util.decorators import vectorize
from .._util.information_theory import marginalize_bits

T = TypeVar("T", np.floating, np.complexfloating)


class Modulation(ABC, Generic[T]):
    @final
    def _validate_parameters(self) -> None:
        order, m = self.order, self.bits_per_symbol
        if order & (order - 1):
            raise ValueError("length of 'constellation' must be a power of two")
        if self.labeling.shape != (order, m):
            raise ValueError(f"shape of 'labeling' must be ({order}, {m})")
        if np.any(self.labeling < 0) or np.any(self.labeling > 1):
            raise ValueError("elements of 'labeling' must be either 0 or 1")
        if len(set(tuple(row) for row in self.labeling)) != order:
            raise ValueError("rows of 'labeling' must be distinct")

    @cached_property
    @abstractmethod
    def constellation(self) -> npt.NDArray[T]:
        r"""
        The constellation $\mathbf{X}$ of the modulation.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        The labeling $\mathbf{Q}$ of the modulation.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        The inverse labeling of the modulation. It is a dictionary that maps each binary tuple to the corresponding constellation index.
        """
        return {tuple(int(x) for x in row): i for i, row in enumerate(self.labeling)}

    @cached_property
    @abstractmethod
    def order(self) -> int:
        r"""
        The order $M$ of the modulation.
        """
        return self.constellation.size

    @cached_property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        r"""
        The number $m$ of bits per symbol of the modulation. It is given by
        $$
            m = \log_2 M,
        $$
        where $M$ is the order of the modulation.
        """
        return (self.order - 1).bit_length()

    @cached_property
    @abstractmethod
    def energy_per_symbol(self) -> float:
        r"""
        The average symbol energy $E_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
          E_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} \lVert x_i \rVert^2,
        $$
        where $\lVert x_i \rVert^2$ is the energy of constellation symbol $x_i$, and $M$ is the order of the modulation.
        """
        return float(np.mean(np.real(self.constellation * self.constellation.conj())))

    @cached_property
    @abstractmethod
    def energy_per_bit(self) -> float:
        r"""
        The average bit energy $E_\mathrm{b}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
            E_\mathrm{b} = \frac{E_\mathrm{s}}{m},
        $$
        where $E_\mathrm{s}$ is the average symbol energy, and $m$ is the number of bits per symbol of the modulation.
        """
        return self.energy_per_symbol / self.bits_per_symbol

    @cached_property
    @abstractmethod
    def symbol_mean(self) -> float | complex:
        r"""
        The mean $\mu_\mathrm{s}$ of the constellation. It assumes equiprobable symbols. It is given by
        $$
          \mu_\mathrm{s} = \frac{1}{M} \sum_{i \in [0:M)} x_i.
        $$
        """
        mean = self.constellation.mean()
        return complex(mean) if np.iscomplexobj(self.constellation) else float(mean)

    @cached_property
    @abstractmethod
    def minimum_distance(self) -> float:
        r"""
        The minimum Euclidean distance $d_\mathrm{min}$ of the constellation. It is given by
        $$
            d_\mathrm{min} = \min_ { i, j \in [0:M), ~ i \neq j } \lVert x_i - x_j \rVert.
        $$
        """
        return float(
            np.min([np.abs(s1 - s2) for s1, s2 in combinations(self.constellation, 2)])
        )

    @abstractmethod
    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Modulates one or more sequences of bits to their corresponding constellation symbols.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $m$, or a multidimensional array where the last dimension is a multiple of $m$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension divided by $m$.
        """

        @vectorize
        def modulate(bits: npt.NDArray[np.integer]) -> npt.NDArray[T]:
            return self.constellation[self.inverse_labeling[tuple(bits)]]

        input = np.asarray(input)
        if input.shape[-1] % self.bits_per_symbol != 0:
            raise ValueError(
                "last dimension of 'bits' must be a multiple of bits per symbol"
                f" {self.bits_per_symbol} (got {input.shape[-1]})"
            )
        bits = input.reshape(*input.shape[:-1], -1, self.bits_per_symbol)
        symbols = modulate(bits)
        output = symbols.reshape(*symbols.shape[:-1], -1)
        return output

    @abstractmethod
    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Demodulates one or more sequences of received points to their corresponding sequences of hard bits ($\mathtt{0}$ or $\mathtt{1}$) using hard-decision decoding.

        Parameters:
            input: The input sequence(s). Can be either a single sequence, or a multidimensional array.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension multiplied by $m$.
        """

        # General minimum Euclidean distance hard demodulator.
        @vectorize
        def demodulate_hard(received: npt.NDArray[T]) -> npt.NDArray[np.integer]:
            return self.labeling[np.argmin(np.abs(self.constellation - received[0]))]

        input = np.asarray(input)
        received = input.reshape(*input.shape[:-1], -1, 1)
        hard_bits = demodulate_hard(received)
        output = hard_bits.reshape(*input.shape[:-1], -1)
        return output

    @abstractmethod
    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        r"""
        Demodulates one or more sequences of received points to their corresponding sequences of soft bits (L-values) using soft-decision decoding. The soft bits are the log-likelihood ratios of the bits, where positive values correspond to bit $\mathtt{0}$ and negative values correspond to bit $\mathtt{1}$.

        Parameters:
            input: The received sequence(s). Can be either a single sequence, or a multidimensional array.
            snr: The signal-to-noise ratio (SNR) of the channel. It should be a positive real number. The default value is `1.0`.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension multiplied by $m$.
        """

        @vectorize
        def demodulate_soft(received: npt.NDArray[T]) -> npt.NDArray[np.floating]:
            # Computes the L-values (LLR) of each bit. Assumes uniformly distributed bits.
            # See [SA15, eq. (3.50)].
            n0 = self.energy_per_symbol / snr
            received = received.ravel()
            distances = np.abs(received - self.constellation) ** 2
            return marginalize_bits(np.exp(-distances / n0), self.labeling)

        input = np.asarray(input)
        received = input.reshape(*input.shape[:-1], -1, 1)
        soft_bits = demodulate_soft(received)
        output = soft_bits.reshape(*input.shape[:-1], -1)
        return output
