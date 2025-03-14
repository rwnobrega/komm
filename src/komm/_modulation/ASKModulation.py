from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base
from .constellations import constellation_ask
from .labelings import get_labeling


class ASKModulation(base.Modulation[np.complexfloating]):
    r"""
    Amplitude-shift keying (ASK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the points of the constellation symbols are *uniformly arranged* in a ray. More precisely, the $i$-th constellation symbol is given by
    $$
        x_i = iA \exp(\mathrm{j}\phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *base amplitude*, and $\phi$ is the *phase offset* of the modulation.

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        1. The $4$-ASK modulation with base amplitude $A = 1$, phase offset $\phi = 0$, and Gray labeling is depicted below.
            <figure markdown>
            ![4-ASK modulation with Gray labeling.](/figures/ask_4_gray.svg)
            </figure>

                >>> ask = komm.ASKModulation(4)
                >>> ask.constellation
                array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
                >>> ask.labeling
                array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]])

        2. The $4$-ASK modulation with base amplitude $A = 2\sqrt{2}$, phase offset $\phi = \pi/4$, and natural labeling is depicted below.
            <figure markdown>
            ![4-ASK modulation with natural labeling.](/figures/ask_4_natural.svg)
            </figure>

                >>> ask = komm.ASKModulation(
                ...     order=4,
                ...     base_amplitude=2*np.sqrt(2),
                ...     phase_offset=np.pi/4,
                ...     labeling='natural',
                ... )
                >>> ask.constellation
                array([0.+0.j, 2.+2.j, 4.+4.j, 6.+6.j])
                >>> ask.labeling
                array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    """

    def __init__(
        self,
        order: int,
        base_amplitude: float = 1.0,
        phase_offset: float = 0.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        self._order = order
        self._base_amplitude = base_amplitude
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
            f"base_amplitude={self._base_amplitude}",
            f"phase_offset={self._phase_offset}",
            f"labeling={labeling_repr}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def constellation(self) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.constellation
            array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
        """
        return constellation_ask(self._order, self._base_amplitude, self._phase_offset)

    @cached_property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.labeling
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
            >>> ask = komm.ASKModulation(4)
            >>> ask.inverse_labeling
            {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        """
        return super().inverse_labeling

    @cached_property
    def order(self) -> int:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.order
            4
        """
        return self._order

    @cached_property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.bits_per_symbol
            2
        """
        return super().bits_per_symbol

    @cached_property
    def energy_per_symbol(self) -> float:
        r"""
        For the ASK, it is given by
        $$
            E_\mathrm{s} = \frac{A^2}{6} (M - 1) (2M - 1).
        $$

        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.energy_per_symbol
            3.5
        """
        return self._base_amplitude**2 * (self._order - 1) * (2 * self._order - 1) / 6

    @cached_property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.energy_per_bit
            1.75
        """
        return super().energy_per_bit

    @cached_property
    def symbol_mean(self) -> complex:
        r"""
        For the ASK, it is given by
        $$
            \mu_\mathrm{s} = \frac{A}{2} (M-1) \exp(\mathrm{j}\phi).
        $$

        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.symbol_mean
            (1.5+0j)
        """
        M, A, phi = self._order, self._base_amplitude, self._phase_offset
        return complex(0.5 * A * (M - 1) * np.exp(1j * phi))

    @cached_property
    def minimum_distance(self) -> float:
        r"""
        For the ASK, it is given by
        $$
            d_\mathrm{min} = A.
        $$

        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.minimum_distance
            1.0
        """
        return self._base_amplitude

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> ask = komm.ASKModulation(4)
            >>> ask.modulate([0, 0, 1, 1, 0, 0, 0, 1])
            array([0.+0.j, 2.+0.j, 0.+0.j, 1.+0.j])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        M, A, phi = self._order, self._base_amplitude, self._phase_offset
        input = np.real(np.multiply(input, np.exp(-1j * phi))) / A
        indices = np.clip(np.round(input).astype(int), 0, M - 1)
        hard_bits = np.reshape(self.labeling[indices], shape=-1)
        return hard_bits

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        return super().demodulate_soft(input, snr)
