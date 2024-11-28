from typing import Literal

import numpy.typing as npt

from .constellations import constellation_psk
from .labelings import get_labeling
from .Modulation import Modulation


class PSKModulation(Modulation):
    r"""
    Phase-shift keying (PSK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the constellation symbols are *uniformly arranged* in a circle. More precisely, the the $i$-th constellation symbol is given by
    $$
        x_i = A \exp \left( \mathrm{j} \frac{2 \pi i}{M} \right) \exp(\mathrm{j} \phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *amplitude*, and $\phi$ is the *phase offset* of the modulation. The PSK constellation is depicted below for $M = 8$.

    <figure markdown>
      ![8-PSK constellation.](/figures/psk_8.svg)
    </figure>

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        amplitude: The amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        The PSK modulation with order $M = 4$, base amplitude $A = 1$, phase offset $\phi = \pi/4$, and Gray labeling is depicted below.

        <figure markdown>
          ![4-PSK modulation with Gray labeling.](/figures/psk_4_gray.svg)
        </figure>

        >>> psk = komm.PSKModulation(4, phase_offset=np.pi/4.0)
        >>> psk.constellation  # doctest: +NORMALIZE_WHITESPACE
        array([ 0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678-0.70710678j])
        >>> psk.labeling
        array([[0, 0],
               [1, 0],
               [1, 1],
               [0, 1]])
        >>> psk.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])  # doctest: +NORMALIZE_WHITESPACE
        array([ 0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678+0.70710678j])
    """

    def __init__(
        self,
        order: int,
        amplitude: float = 1.0,
        phase_offset: float = 0.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        super().__init__(
            constellation=constellation_psk(order, amplitude, phase_offset),
            labeling=get_labeling(labeling, ("natural", "reflected"), order),
        )
        self.amplitude = amplitude
        self.phase_offset = phase_offset
        self.labeling_parameter = labeling

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self.order}",
            f"amplitude={self.amplitude}",
            f"phase_offset={self.phase_offset}",
            f"labeling='{self.labeling_parameter}'",
        ])
        return f"PSKModulation({args})"
