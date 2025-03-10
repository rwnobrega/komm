from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._util.decorators import vectorize
from .._util.special_functions import logcosh
from . import base
from .constellations import constellation_pam
from .labelings import get_labeling


class PAModulation(base.Modulation[np.floating]):
    r"""
    Pulse-amplitude modulation (PAM). It is a real [modulation scheme](/ref/Modulation) in which the constellation symbols are *uniformly arranged* in the real line and have zero mean. More precisely, the the $i$-th constellation symbol is given by
    $$
        x_i = A \left( 2i - M + 1 \right), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), and $A$ is the *base amplitude* of the modulation.

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        1. The PAM modulation with order $M = 4$, base amplitude $A = 1$, and natural labeling is depicted below.
            <figure markdown>
            ![4-PAM modulation with Gray labeling.](/figures/pam_4_natural.svg)
            </figure>

                >>> pam = komm.PAModulation(4, labeling="natural")
                >>> pam.constellation
                array([-3., -1.,  1.,  3.])
                >>> pam.labeling
                array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1]])

        1. The PAM modulation with order $M = 8$, base amplitude $A = 0.5$, and Gray labeling is depicted below.
            <figure markdown>
            ![8-PAM modulation with Gray labeling.](/figures/pam_8_gray.svg)
            </figure>

                >>> pam = komm.PAModulation(8, base_amplitude=0.5)
                >>> pam.constellation
                array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
                >>> pam.labeling
                array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [0, 0, 1]])
    """

    def __init__(
        self,
        order: int,
        base_amplitude: float = 1.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        self._order = order
        self._constellation = constellation_pam(order, base_amplitude)
        self._labeling_parameter = labeling
        self._labeling = get_labeling(labeling, ("natural", "reflected"), order)
        self._base_amplitude = base_amplitude
        super()._validate_parameters()

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self._order}",
            f"base_amplitude={self._base_amplitude}",
            f"labeling='{self._labeling_parameter}'",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    def constellation(self) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.constellation
            array([-3., -1.,  1.,  3.])
        """
        return self._constellation

    @property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.labeling
            array([[0, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1]])
        """
        return self._labeling

    @cached_property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.inverse_labeling
            {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}
        """
        return super().inverse_labeling

    @property
    def order(self) -> int:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.order
            4
        """
        return self._order

    @property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.bits_per_symbol
            2
        """
        return super().bits_per_symbol

    @property
    def energy_per_symbol(self) -> float:
        r"""
        For the PAM, it is given by
        $$
            E_\mathrm{s} = \frac{A^2}{3} \left( M^2 - 1 \right)\vphantom{\Bigg|}.
        $$

        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.energy_per_symbol
            5.0
        """
        return (self._base_amplitude**2) * (self._order**2 - 1) / 3

    @property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.energy_per_bit
            2.5
        """
        return super().energy_per_bit

    @property
    def symbol_mean(self) -> float:
        r"""
        For the PAM, it is given by
        $$
            \mu_\mathrm{s} = 0.
        $$

        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.symbol_mean
            0.0
        """
        return 0.0

    @property
    def minimum_distance(self) -> float:
        r"""
        For the PAM, it is given by
        $$
            d_\mathrm{min} = 2A.
        $$

        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.minimum_distance
            2.0
        """
        return 2.0 * self._base_amplitude

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> pam = komm.PAModulation(4)
            >>> pam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
            array([-3.,  1., -3., -1., -1.])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        input = np.asarray(input)
        indices = np.clip(
            np.around((input + self._order - 1) / 2), 0, self._order - 1
        ).astype(int)
        hard_bits = np.reshape(self.labeling[indices], shape=-1)
        return hard_bits

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        if self._order == 2:
            y = np.asarray(input) / self._base_amplitude
            return self._demodulate_pam2_soft(y, snr / 1.0)
        elif self._order == 4 and self._labeling_parameter == "reflected":
            y = np.asarray(input) / self._base_amplitude
            return vectorize(self._demodulate_pam4_soft_reflected)(y, snr / 5.0)  # type: ignore
        # Fall back to general implementation
        return super().demodulate_soft(input, snr)

    def _demodulate_pam2_soft(self, y: npt.NDArray[np.floating], gamma: float):
        return -4 * gamma * y  # [SA15, eq. (3.65)]

    def _demodulate_pam4_soft_reflected(
        self, y: npt.NDArray[np.floating], gamma: float
    ):
        soft_bits = np.empty(2 * y.size, dtype=float)
        soft_bits[0::2] = (  # For bit_0: [SA15, eq. (5.15)]
            -8 * gamma + logcosh(6 * gamma * y) - logcosh(2 * gamma * y)
        )
        soft_bits[1::2] = (  # For bit_1: [SA15, eq. (5.6)]
            -8 * gamma * y + logcosh(2 * gamma * (y + 2)) - logcosh(2 * gamma * (y - 2))
        )
        return soft_bits
