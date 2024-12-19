import math
from collections.abc import Iterable
from typing import Literal

import numpy.typing as npt

from .constellations import constellation_qam
from .labelings import get_labeling
from .Modulation import Modulation


class QAModulation(Modulation):
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
    where $M_\mathrm{I}$ and $M_\mathrm{Q}$ are the *orders* (powers of $2$), and $A_\mathrm{I}$ and $A_\mathrm{Q}$ are the *base amplitudes* of the in-phase and quadrature constellations, respectively. Also, $\phi$ is the *phase offset*. The order of the resulting complex-valued constellation is $M = M_\mathrm{I} M_\mathrm{Q}$, a power of $2$. The QAM constellation is depicted below for $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ with ($A_\mathrm{I}, A_\mathrm{Q}) = (A, A)$; and for $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 2)$ with $(A_\mathrm{I}, A_\mathrm{Q}) = (A, 2A)$; in both cases, $\phi = 0$.

    <div class="centered" markdown>
      <span>
        ![16-QAM constellation.](/figures/qam_16.svg)
      </span>
      <span>
        ![8-QAM constellation.](/figures/qam_8.svg)
      </span>
    </div>

    Parameters:
        orders: A tuple $(M_\mathrm{I}, M_\mathrm{Q})$ with the orders of the in-phase and quadrature constellations, respectively; both $M_\mathrm{I}$ and $M_\mathrm{Q}$ must be powers of $2$. If specified as a single integer $M$, then it is assumed that $M_\mathrm{I} = M_\mathrm{Q} = \sqrt{M}$; in this case, $M$ must be an square power of $2$.

        base_amplitudes: A tuple $(A_\mathrm{I}, A_\mathrm{Q})$ with the base amplitudes of the in-phase and quadrature constellations, respectively. If specified as a single float $A$, then it is assumed that $A_\mathrm{I} = A_\mathrm{Q} = A$. The default value is $1.0$.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural_2d'` or `'reflected_2d'`. The default value is `'reflected_2d'`, corresponding to the Gray labeling.

    Examples:
        The square $16$-QAM modulation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ and $(A_\mathrm{I}, A_\mathrm{Q}) = (1, 1)$, and Gray labeling is depicted below.

        <figure markdown>
          ![16-QAM modulation with Gray labeling.](/figures/qam_16_gray.svg)
        </figure>

        >>> qam = komm.QAModulation(16)
        >>> qam.constellation  # doctest: +NORMALIZE_WHITESPACE
        array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
               -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
               -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
               -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
        >>> qam.labeling   # doctest: +NORMALIZE_WHITESPACE
        array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0],
               [0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0],
               [0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1],
               [0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]])
        >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0])
        array([-3.+1.j, -3.-1.j])

        The rectangular $8$-QAM modulation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 2)$ and $(A_\mathrm{I}, A_\mathrm{Q}) = (1, 2)$, and Gray labeling is depicted below.

        <figure markdown>
          ![8-QAM modulation with Gray labeling.](/figures/qam_8_gray.svg)
        </figure>

        >>> qam = komm.QAModulation(orders=(4, 2), base_amplitudes=(1.0, 2.0))
        >>> qam.constellation  # doctest: +NORMALIZE_WHITESPACE
        array([-3.-2.j, -1.-2.j,  1.-2.j,  3.-2.j,
               -3.+2.j, -1.+2.j,  1.+2.j,  3.+2.j])
        >>> qam.labeling  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
               [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1])
        array([-3.+2.j, -1.-2.j, -1.+2.j])
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
        _orders = (
            orders
            if isinstance(orders, Iterable)
            else (math.isqrt(orders), math.isqrt(orders))
        )
        _base_amplitudes = (
            base_amplitudes
            if isinstance(base_amplitudes, Iterable)
            else (base_amplitudes, base_amplitudes)
        )
        super().__init__(
            constellation=constellation_qam(_orders, _base_amplitudes, phase_offset),
            labeling=get_labeling(labeling, ("natural_2d", "reflected_2d"), _orders),
        )
        self.orders = orders
        self.base_amplitudes = base_amplitudes
        self.phase_offset = phase_offset
        self.labeling_parameter = labeling

    def __repr__(self) -> str:
        args = ", ".join([
            f"orders={self.orders}",
            f"base_amplitudes={self.base_amplitudes}",
            f"phase_offset={self.phase_offset}",
            f"labeling='{self.labeling_parameter}'",
        ])
        return f"{self.__class__.__name__}({args})"
