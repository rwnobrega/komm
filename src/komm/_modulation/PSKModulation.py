from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base
from .constellations import constellation_psk
from .labelings import get_labeling


class PSKModulation(base.Modulation[np.complexfloating]):
    r"""
    Phase-shift keying (PSK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the constellation symbols are *uniformly arranged* in a circle. More precisely, the the $i$-th constellation symbol is given by
    $$
        x_i = A \exp \left( \mathrm{j} \frac{2 \pi i}{M} \right) \exp(\mathrm{j} \phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *amplitude*, and $\phi$ is the *phase offset* of the modulation.

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        amplitude: The amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        1. The $4$-PSK modulation with base amplitude $A = 1$, phase offset $\phi = 0$, and Gray labeling is depicted below.
            <figure markdown>
            ![4-PSK modulation with Gray labeling.](/figures/psk_4_gray.svg)
            </figure>

                >>> psk = komm.PSKModulation(4)
                >>> psk.constellation.round(3)
                array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j])
                >>> psk.labeling
                array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]])

        1. The $8$-PSK modulation with base amplitude $A = 0.5$, phase offset $\phi = \pi/8$, and natural labeling is depicted below.
            <figure markdown>
            ![8-PSK modulation with natural labeling.](/figures/psk_8_natural.svg)
            </figure>

                >>> psk = komm.PSKModulation(
                ...     order=8,
                ...     amplitude=0.5,
                ...     phase_offset=np.pi/8,
                ...     labeling='natural',
                ... )
                >>> psk.constellation.round(3)
                array([ 0.462+0.191j,  0.191+0.462j, -0.191+0.462j, -0.462+0.191j,
                       -0.462-0.191j, -0.191-0.462j,  0.191-0.462j,  0.462-0.191j])
                >>> psk.labeling
                array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
    """

    def __init__(
        self,
        order: int,
        amplitude: float = 1.0,
        phase_offset: float = 0.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        self._order = order
        self._amplitude = amplitude
        self._phase_offset = phase_offset
        self._labeling = labeling
        super()._validate_parameters()

    def __repr__(self) -> str:
        if isinstance(self._labeling, str):
            labeling_repr = repr(self._labeling)
        else:
            labeling_repr = self.labeling.tolist()
        args = ", ".join([
            f"order={self._order}",
            f"amplitude={self._amplitude}",
            f"phase_offset={self._phase_offset}",
            f"labeling={labeling_repr}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def constellation(self) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.constellation.round(3)
            array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j])
        """
        return constellation_psk(self._order, self._amplitude, self._phase_offset)

    @cached_property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.labeling
            array([[0, 0],
                   [0, 1],
                   [1, 1],
                   [1, 0]])
        """
        return get_labeling(self._labeling, ("natural", "reflected"), self._order)

    @cached_property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.inverse_labeling
            {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        """
        return super().inverse_labeling

    @cached_property
    def order(self) -> int:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.order
            4
        """
        return self._order

    @cached_property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.bits_per_symbol
            2
        """
        return super().bits_per_symbol

    @cached_property
    def energy_per_symbol(self) -> float:
        r"""
        For the PSK, it is given by
        $$
            E_\mathrm{s} = A^2.
        $$

        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.energy_per_symbol
            1.0
        """
        return self._amplitude**2

    @cached_property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.energy_per_bit
            0.5
        """
        return super().energy_per_bit

    @cached_property
    def symbol_mean(self) -> complex:
        r"""
        For the PSK, it is given by
        $$
            \mu_\mathrm{s} = 0.
        $$

        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.symbol_mean
            0j
        """
        return 0j

    @cached_property
    def minimum_distance(self) -> float:
        r"""
        For the PSK, it is given by
        $$
            d_\mathrm{min} = 2A\sin\left(\frac{\pi}{M}\right).
        $$

        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.minimum_distance  # doctest: +FLOAT_CMP
            1.414213562373095
        """
        return 2.0 * self._amplitude * float(np.sin(np.pi / self._order))

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> psk = komm.PSKModulation(4)
            >>> psk.modulate([0, 0, 1, 1, 0, 0, 0, 1]).round(3)  # doctest: +FLOAT_CMP
            array([ 1.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        M, phi = self._order, self._phase_offset
        turn = np.angle(np.multiply(input, np.exp(-1j * phi))) / (2 * np.pi)
        indices = np.mod(np.around(M * turn), M).astype(int)
        hard_bits = np.reshape(self.labeling[indices], shape=-1)
        return hard_bits

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        return super().demodulate_soft(input, snr)
