import math
from collections.abc import Iterable
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base
from .constellations import constellation_qam
from .labelings import get_labeling


class QAModulation(base.Modulation[np.complexfloating]):
    r"""
    Quadrature-amplitude modulation (QAM). It is a complex [modulation scheme](/ref/Modulation) in which the constellation is given as a Cartesian product of two [PAM](/ref/PAModulation) constellations, namely, the *in-phase constellation*, and the *quadrature constellation*. More precisely, the $i$-th constellation symbol is given by
    $$
        \begin{aligned}
            x_i = \left[ A_\mathrm{I} \left( 2i_\mathrm{I} - M_\mathrm{I} + 1 \right) + \mathrm{j} A_\mathrm{Q} \left( 2i_\mathrm{Q} - M_\mathrm{Q} + 1 \right) \right] \exp(\mathrm{j}\phi), \quad
                &  i \in [0 : M), \\\\
                & i_\mathrm{I} = i \bmod M_\mathrm{I}, \\\\
                & i_\mathrm{Q} = \lfloor i / M_\mathrm{I} \rfloor,
        \end{aligned}
    $$
    where $M_\mathrm{I}$ and $M_\mathrm{Q}$ are the *orders* (powers of $2$), and $A_\mathrm{I}$ and $A_\mathrm{Q}$ are the *base amplitudes* of the in-phase and quadrature constellations, respectively. Also, $\phi$ is the *phase offset*. The order of the resulting complex-valued constellation is $M = M_\mathrm{I} M_\mathrm{Q}$, a power of $2$.

    Parameters:
        orders: A tuple $(M_\mathrm{I}, M_\mathrm{Q})$ with the orders of the in-phase and quadrature constellations, respectively; both $M_\mathrm{I}$ and $M_\mathrm{Q}$ must be powers of $2$. If specified as a single integer $M$, then it is assumed that $M_\mathrm{I} = M_\mathrm{Q} = \sqrt{M}$; in this case, $M$ must be an square power of $2$.

        base_amplitudes: A tuple $(A_\mathrm{I}, A_\mathrm{Q})$ with the base amplitudes of the in-phase and quadrature constellations, respectively. If specified as a single float $A$, then it is assumed that $A_\mathrm{I} = A_\mathrm{Q} = A$. The default value is $1.0$.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural_2d'` or `'reflected_2d'`. The default value is `'reflected_2d'`, corresponding to the Gray labeling.

    Examples:
        1. The square $16$-QAM modulation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ and $(A_\mathrm{I}, A_\mathrm{Q}) = (1, 1)$, and Gray labeling is depicted below.
            <figure markdown>
            ![16-QAM modulation with Gray labeling.](/figures/qam_16_gray.svg)
            </figure>

                >>> qam = komm.QAModulation(16)
                >>> qam.constellation
                array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
                       -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
                       -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
                       -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
                >>> qam.labeling
                array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0],
                       [0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1],
                       [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1],
                       [0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0]])

        1.  The rectangular $8$-QAM modulation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 2)$ and $(A_\mathrm{I}, A_\mathrm{Q}) = (1, 2)$, and natural labeling is depicted below.
            <figure markdown>
            ![8-QAM modulation with Gray labeling.](/figures/qam_8_natural.svg)
            </figure>

                >>> qam = komm.QAModulation(
                ...     orders=(4, 2),
                ...     base_amplitudes=(1.0, 2.0),
                ...     labeling="natural_2d"
                ... )
                >>> qam.constellation
                array([-3.-2.j, -1.-2.j,  1.-2.j,  3.-2.j,
                       -3.+2.j, -1.+2.j,  1.+2.j,  3.+2.j])
                >>> qam.labeling
                array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
                       [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    """

    def __init__(
        self,
        orders: tuple[int, int] | int,
        base_amplitudes: tuple[float, float] | float = 1.0,
        phase_offset: float = 0.0,
        labeling: (
            Literal["natural_2d", "reflected_2d"] | npt.ArrayLike
        ) = "reflected_2d",
    ) -> None:
        if isinstance(orders, Iterable):
            self._orders = orders
        else:
            if not math.isqrt(orders) ** 2 == orders:
                raise ValueError(
                    "when a single integer, 'orders' must be a square power of 2"
                )
            self._orders = (math.isqrt(orders), math.isqrt(orders))
        if isinstance(base_amplitudes, Iterable):
            self._base_amplitudes = base_amplitudes
        else:
            self._base_amplitudes = (base_amplitudes, base_amplitudes)
        self._phase_offset = phase_offset
        self._labeling = labeling
        super()._validate_parameters()

    def __repr__(self) -> str:
        if isinstance(self._labeling, str):
            labeling_repr = repr(self._labeling)
        else:
            labeling_repr = self.labeling.tolist()
        args = ", ".join([
            f"orders={self._orders}",
            f"base_amplitudes={self._base_amplitudes}",
            f"phase_offset={self._phase_offset}",
            f"labeling={labeling_repr}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def constellation(self) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.constellation
            array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
                   -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
                   -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
                   -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
        """
        return constellation_qam(
            self._orders, self._base_amplitudes, self._phase_offset
        )

    @cached_property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.labeling
            array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0],
                   [0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1],
                   [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1],
                   [0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0]])
        """
        return get_labeling(
            self._labeling, ("natural_2d", "reflected_2d"), self._orders
        )

    @cached_property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.inverse_labeling
            {(0, 0, 0, 0): 0,  (0, 1, 0, 0): 1,  (1, 1, 0, 0): 2,  (1, 0, 0, 0): 3,
             (0, 0, 0, 1): 4,  (0, 1, 0, 1): 5,  (1, 1, 0, 1): 6,  (1, 0, 0, 1): 7,
             (0, 0, 1, 1): 8,  (0, 1, 1, 1): 9,  (1, 1, 1, 1): 10, (1, 0, 1, 1): 11,
             (0, 0, 1, 0): 12, (0, 1, 1, 0): 13, (1, 1, 1, 0): 14, (1, 0, 1, 0): 15}
        """
        return super().inverse_labeling

    @cached_property
    def order(self) -> int:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.order
            16
        """
        return self._orders[0] * self._orders[1]

    @cached_property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.bits_per_symbol
            4
        """
        return super().bits_per_symbol

    @cached_property
    def energy_per_symbol(self) -> float:
        r"""
        For the QAM, it is given by
        $$
            E_\mathrm{s} = \frac{A_\mathrm{I}^2}{3} \left( M_\mathrm{I}^2 - 1 \right) + \frac{A_\mathrm{Q}^2}{3} \left( M_\mathrm{Q}^2 - 1 \right).
        $$
        For the special case of a square QAM, it simplifies to
        $$
            E_\mathrm{s} = \frac{2A^2}{3}(M - 1).
        $$

        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.energy_per_symbol
            10.0
        """
        A_I, A_Q = self._base_amplitudes
        M_I, M_Q = self._orders
        return (A_I**2) * (M_I**2 - 1) / 3 + (A_Q**2) * (M_Q**2 - 1) / 3

    @cached_property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.energy_per_bit
            2.5
        """
        return super().energy_per_bit

    @cached_property
    def symbol_mean(self) -> complex:
        r"""
        For the QAM, it is given by
        $$
            \mu_\mathrm{s} = 0.
        $$

        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.symbol_mean
            0j
        """
        return 0j

    @cached_property
    def minimum_distance(self) -> float:
        r"""
        For the QAM, it is given by
        $$
            d_\mathrm{min} = 2 \min(A_\mathrm{I}, A_\mathrm{Q}).
        $$
        For the special case of a square QAM, it simplifies to
        $$
            d_\mathrm{min} = 2 A.
        $$

        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.minimum_distance
            2.0
        """
        return 2.0 * min(self._base_amplitudes)

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> qam = komm.QAModulation(16)
            >>> qam.modulate([0, 0, 1, 1, 0, 0, 0, 1])
            array([-3.+1.j, -3.-1.j])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        A_I, A_Q = self._base_amplitudes
        M_I, M_Q = self._orders
        input_I = np.real(np.multiply(input, np.exp(-1j * self._phase_offset))) / A_I
        input_Q = np.imag(np.multiply(input, np.exp(-1j * self._phase_offset))) / A_Q
        indices_I = np.clip(np.around((input_I + M_I - 1) / 2), 0, M_I - 1).astype(int)
        indices_Q = np.clip(np.around((input_Q + M_Q - 1) / 2), 0, M_Q - 1).astype(int)
        indices = indices_I + indices_Q * M_I
        hard_bits = np.reshape(self.labeling[indices], shape=-1)
        return hard_bits

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        return super().demodulate_soft(input, snr)
