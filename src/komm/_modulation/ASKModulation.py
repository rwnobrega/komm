from typing import Literal

import numpy.typing as npt

from .constellations import constellation_ask
from .labelings import get_labeling
from .Modulation import Modulation


class ASKModulation(Modulation):
    r"""
    Amplitude-shift keying (ASK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the points of the constellation symbols are *uniformly arranged* in a ray. More precisely, the $i$-th constellation symbol is given by
    $$
        x_i = iA \exp(\mathrm{j}\phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *base amplitude*, and $\phi$ is the *phase offset* of the modulation. The ASK constellation is depicted below for $M = 4$.

    <figure markdown>
      ![4-ASK constellation.](/figures/ask_4.svg)
    </figure>

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

        phase_offset: The phase offset $\phi$ of the constellation. The default value is `0.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        The ASK modulation with order $M = 4$, base amplitude $A = 1$, and Gray labeling is depicted below.

        <figure markdown>
          ![4-ASK modulation with Gray labeling.](/figures/ask_4_gray.svg)
        </figure>

        >>> ask = komm.ASKModulation(4)
        >>> ask.constellation
        array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
        >>> ask.labeling
        array([[0, 0],
               [1, 0],
               [1, 1],
               [0, 1]])
        >>> ask.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        array([0.+0.j, 2.+0.j, 0.+0.j, 1.+0.j, 1.+0.j])
    """

    def __init__(
        self,
        order: int,
        base_amplitude: float = 1.0,
        phase_offset: float = 0.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        super().__init__(
            constellation=constellation_ask(order, base_amplitude, phase_offset),
            labeling=get_labeling(labeling, ("natural", "reflected"), order),
        )
        self.base_amplitude = base_amplitude
        self.phase_offset = phase_offset
        self.labeling_parameter = labeling

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self.order}",
            f"base_amplitude={self.base_amplitude}",
            f"phase_offset={self.phase_offset}",
            f"labeling='{self.labeling_parameter}'",
        ])
        return f"{self.__class__.__name__}({args})"
