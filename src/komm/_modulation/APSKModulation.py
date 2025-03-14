from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base
from .constellations import constellation_apsk
from .labelings import get_labeling


class APSKModulation(base.Modulation[np.complexfloating]):
    r"""
    Amplitude- and phase-shift keying (APSK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the constellation is the union (concatenation) of component [PSK](/ref/PSKModulation) constellations, called *rings*. More precisely, consider $K$ rings $\mathbf{X}_k$, for $k \in [0 : K)$, where the $k$-th ring has order $M_k$, amplitude $A_k$, and phase offset $\phi_k$. The $i$-th constellation symbol of the $k$-th ring is given by
    $$
        x\_{k,i} = A_k \exp \left( \mathrm{j} \frac{2 \pi i}{M_k} \right) \exp(\mathrm{j} \phi_k),
        \quad k \in [0 : K),
        \quad i \in [0 : M_k).
    $$
    The resulting APSK constellation is therefore given by
    $$
        \mathbf{X} = \begin{bmatrix}
            \mathbf{X}_0 \\\\
            \vdots \\\\
            \mathbf{X}\_{K-1}
        \end{bmatrix},
    $$
    which has order $M = M_0 + M_1 + \cdots + M\_{K-1}$. The order $M_k$ of each ring need not be a power of $2$; however, the order $M$ of the constructed APSK modulation must be.

    Parameters:
        orders: A $K$-tuple with the orders $M_k$ of each ring, for $k \in [0 : K)$. The sum $M_0 + M_1 + \cdots + M_{K-1}$ must be a power of $2$.

        amplitudes: A $K$-tuple with the amplitudes $A_k$ of each ring, for $k \in [0 : K)$.

        phase_offsets: A $K$-tuple with the phase offsets $\phi_k$ of each ring, for $k \in [0 : K)$. If specified as a single float $\phi$, then it is assumed that $\phi_k = \phi$ for all $k \in [0 : K)$. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be equal to `'natural'`. The default value is `'natural'`.

    Examples:

    1. The $8$-APSK modulation with $(M_0, M_1) = (4, 4)$, $(A_0, A_1) = (1, 2)$, and $(\phi_0, \phi_1) = (0, 0)$ is depicted below.
        <figure markdown>
        ![(4,4)-APSK modulation.](/figures/apsk_4_4.svg)
        </figure>

            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.constellation.round(3)
            array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j,
                    2.+0.j,  0.+2.j, -2.+0.j, -0.-2.j])

    1. The $16$-APSK modulation with $(M_0, M_1) = (8, 8)$, $(A_0, A_1) = (1, 2)$, and $(\phi_0, \phi_1) = (0, \pi/8)$ is depicted below.
        <figure markdown>
        ![(8,8)-APSK modulation.](/figures/apsk_8_8.svg)
        </figure>

            >>> apsk = komm.APSKModulation(
            ...     orders=(8, 8),
            ...     amplitudes=(1.0, 2.0),
            ...     phase_offsets=(0.0, np.pi/8)
            ... )
            >>> apsk.constellation.round(3)
            array([ 1.   +0.j   ,  0.707+0.707j,  0.   +1.j   , -0.707+0.707j,
                   -1.   +0.j   , -0.707-0.707j, -0.   -1.j   ,  0.707-0.707j,
                    1.848+0.765j,  0.765+1.848j, -0.765+1.848j, -1.848+0.765j,
                   -1.848-0.765j, -0.765-1.848j,  0.765-1.848j,  1.848-0.765j])

    1. The $16$-APSK modulation with $(M_0, M_1) = (4, 12)$, $(A_0, A_1) = (\sqrt{2}, 3)$, and $(\phi_0, \phi_1) = (\pi/4, 0)$ is depicted below.
        <figure markdown>
        ![(4,12)-APSK modulation.](/figures/apsk_4_12.svg)
        </figure>

            >>> apsk = komm.APSKModulation(
            ...     orders=(4, 12),
            ...     amplitudes=(np.sqrt(2), 3.0),
            ...     phase_offsets=(np.pi/4, 0.0)
            ... )
            >>> apsk.constellation.round(3)
            array([ 1.   +1.j   , -1.   +1.j   , -1.   -1.j   ,  1.   -1.j   ,
                    3.   +0.j   ,  2.598+1.5j  ,  1.5  +2.598j,  0.   +3.j   ,
                   -1.5  +2.598j, -2.598+1.5j  , -3.   +0.j   , -2.598-1.5j  ,
                   -1.5  -2.598j, -0.   -3.j   ,  1.5  -2.598j,  2.598-1.5j  ])
    """

    def __init__(
        self,
        orders: tuple[int, ...],
        amplitudes: tuple[float, ...],
        phase_offsets: float | tuple[float, ...] = 0.0,
        labeling: Literal["natural"] | npt.ArrayLike = "natural",
    ) -> None:
        self._orders = orders
        self._amplitudes = amplitudes
        if isinstance(phase_offsets, tuple):
            self._phase_offsets = phase_offsets
        else:
            self._phase_offsets = (phase_offsets,) * len(orders)
        self._labeling = labeling
        super()._validate_parameters()

    def __repr__(self) -> str:
        if isinstance(self._labeling, str):
            labeling_repr = repr(self._labeling)
        else:
            labeling_repr = self.labeling.tolist()
        args = ", ".join([
            f"orders={self._orders}",
            f"amplitudes={self._amplitudes}",
            f"phase_offsets={self._phase_offsets}",
            f"labeling={labeling_repr}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def constellation(self) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.constellation.round(3)
            array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j,
                    2.+0.j,  0.+2.j, -2.+0.j, -0.-2.j])
        """
        return constellation_apsk(self._orders, self._amplitudes, self._phase_offsets)

    @cached_property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.labeling
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]])
        """
        return get_labeling(self._labeling, ("natural",), sum(self._orders))

    @cached_property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.inverse_labeling
            {(0, 0, 0): 0,
             (0, 0, 1): 1,
             (0, 1, 0): 2,
             (0, 1, 1): 3,
             (1, 0, 0): 4,
             (1, 0, 1): 5,
             (1, 1, 0): 6,
             (1, 1, 1): 7}
        """
        return super().inverse_labeling

    @cached_property
    def order(self) -> int:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.order
            8
        """
        return sum(self._orders)

    @cached_property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.bits_per_symbol
            3
        """
        return super().bits_per_symbol

    @cached_property
    def energy_per_symbol(self) -> float:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.energy_per_symbol
            2.5
        """
        return super().energy_per_symbol

    @cached_property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.energy_per_bit
            0.8333333333333334
        """
        return super().energy_per_bit

    @cached_property
    def symbol_mean(self) -> complex:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.symbol_mean
            0j
        """
        return 0j

    @cached_property
    def minimum_distance(self) -> float:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.minimum_distance
            1.0
        """
        return super().minimum_distance

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> apsk = komm.APSKModulation(orders=(4, 4), amplitudes=(1.0, 2.0))
            >>> apsk.modulate([0, 0, 0, 0, 1, 1, 0, 0, 0]).round(3)
            array([ 1.+0.j, -0.-1.j,  1.+0.j])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().demodulate_hard(input)

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        return super().demodulate_soft(input, snr)
