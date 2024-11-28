from typing import Literal

import numpy.typing as npt

from .constellations import constellation_apsk
from .labelings import get_labeling
from .Modulation import Modulation


class APSKModulation(Modulation):
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
    which has order $M = M_0 + M_1 + \cdots + M\_{K-1}$. The order $M_k$ of each ring need not be a power of $2$; however, the order $M$ of the constructed APSK modulation must be. The APSK constellation is depicted below for $(M_0, M_1) = (8, 8)$ with $(A_0, A_1) = (A, 2A)$ and $(\phi_0, \phi_1) = (0, \pi/8)$; and for $(M_0, M_1) = (4, 12)$ with $(A_0, A_1) = (\sqrt{2}A, 3A)$ and $(\phi_0, \phi_1) = (\pi/4, 0)$.

    <div class="centered" markdown>
      <span>
        ![(8,8)-APSK constellation.](/figures/apsk_8_8.svg)
      </span>
      <span>
        ![(4,12)-APSK constellation.](/figures/apsk_4_12.svg)
      </span>
    </div>

    Parameters:
        orders: A $K$-tuple with the orders $M_k$ of each ring, for $k \in [0 : K)$. The sum $M_0 + M_1 + \cdots + M_{K-1}$ must be a power of $2$.

        amplitudes: A $K$-tuple with the amplitudes $A_k$ of each ring, for $k \in [0 : K)$.

        phase_offsets: A $K$-tuple with the phase offsets $\phi_k$ of each ring, for $k \in [0 : K)$. If specified as a single float $\phi$, then it is assumed that $\phi_k = \phi$ for all $k \in [0 : K)$. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be equal to `'natural'`. The default value is `'natural'`.

    Examples:
        >>> apsk = komm.APSKModulation(orders=(8, 8), amplitudes=(1.0, 2.0), phase_offsets=(0.0, np.pi/8))
        >>> apsk.constellation.round(4)
        array([ 1.    +0.j    ,  0.7071+0.7071j,  0.    +1.j    , -0.7071+0.7071j,
               -1.    +0.j    , -0.7071-0.7071j, -0.    -1.j    ,  0.7071-0.7071j,
                1.8478+0.7654j,  0.7654+1.8478j, -0.7654+1.8478j, -1.8478+0.7654j,
               -1.8478-0.7654j, -0.7654-1.8478j,  0.7654-1.8478j,  1.8478-0.7654j])

        >>> apsk = komm.APSKModulation(orders=(4, 12), amplitudes=(np.sqrt(2), 3.0), phase_offsets=(np.pi/4, 0.0))
        >>> apsk.constellation.round(4)
        array([ 1.    +1.j    , -1.    +1.j    , -1.    -1.j    ,  1.    -1.j    ,
                3.    +0.j    ,  2.5981+1.5j   ,  1.5   +2.5981j,  0.    +3.j    ,
               -1.5   +2.5981j, -2.5981+1.5j   , -3.    +0.j    , -2.5981-1.5j   ,
               -1.5   -2.5981j, -0.    -3.j    ,  1.5   -2.5981j,  2.5981-1.5j   ])
    """

    def __init__(
        self,
        orders: tuple[int, ...],
        amplitudes: tuple[float, ...],
        phase_offsets: float | tuple[float, ...] = 0.0,
        labeling: Literal["natural"] | npt.ArrayLike = "natural",
    ) -> None:
        _phase_offsets = (
            phase_offsets
            if isinstance(phase_offsets, tuple)
            else (phase_offsets,) * len(orders)
        )
        super().__init__(
            constellation=constellation_apsk(orders, amplitudes, _phase_offsets),
            labeling=get_labeling(labeling, ("natural",), sum(orders)),
        )
        self.orders = orders
        self.amplitudes = amplitudes
        self.phase_offsets = phase_offsets
        self.labeling_parameter = labeling

    def __repr__(self) -> str:
        args = ", ".join([
            f"orders={self.orders}",
            f"amplitudes={self.amplitudes}",
            f"phase_offsets={self.phase_offsets}",
            f"labeling='{self.labeling_parameter}'",
        ])
        return f"{self.__class__.__name__}({args})"
