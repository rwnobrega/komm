from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from .._util.decorators import vectorize
from .._util.special_functions import logcosh
from .constellations import constellation_pam
from .labelings import get_labeling
from .Modulation import Modulation


class PAModulation(Modulation):
    r"""
    Pulse-amplitude modulation (PAM). It is a real [modulation scheme](/ref/Modulation) in which the constellation symbols are *uniformly arranged* in the real line and have zero mean. More precisely, the the $i$-th constellation symbol is given by
    $$
        x_i = A \left( 2i - M + 1 \right), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), and $A$ is the *base amplitude* of the modulation. The PAM constellation is depicted below for $M = 8$.

    <figure markdown>
      ![8-PAM constellation.](/figures/pam_8.svg)
    </figure>

    Parameters:
        order: The order $M$ of the modulation. It must be a power of $2$.

        base_amplitude: The base amplitude $A$ of the constellation. The default value is `1.0`.

        labeling: The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

    Examples:
        The PAM modulation with order $M = 4$, base amplitude $A = 1$, and Gray labeling is depicted below.

        <figure markdown>
          ![4-PAM modulation with Gray labeling.](/figures/pam_4_gray.svg)
        </figure>

        >>> pam = komm.PAModulation(4)
        >>> pam.constellation
        array([-3., -1.,  1.,  3.])
        >>> pam.labeling
        array([[0, 0],
               [1, 0],
               [1, 1],
               [0, 1]])
        >>> pam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        array([-3.,  1., -3., -1., -1.])
    """

    def __init__(
        self,
        order: int,
        base_amplitude: float = 1.0,
        labeling: Literal["natural", "reflected"] | npt.ArrayLike = "reflected",
    ) -> None:
        super().__init__(
            constellation=constellation_pam(order, base_amplitude),
            labeling=get_labeling(labeling, ("natural", "reflected"), order),
        )
        self.base_amplitude = base_amplitude
        self.labeling_parameter = labeling

    def __repr__(self) -> str:
        args = ", ".join([
            f"order={self.order}",
            f"base_amplitude={self.base_amplitude}",
            f"labeling='{self.labeling_parameter}'",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    @override
    def energy_per_symbol(self) -> float:
        return (self.base_amplitude**2) * (self.order**2 - 1) / 3

    @property
    @override
    def symbol_mean(self) -> float:
        return 0.0

    @property
    @override
    def minimum_distance(self) -> float:
        return 2.0 * self.base_amplitude

    @override
    def _demodulate_hard(
        self, received: npt.NDArray[np.floating | np.complexfloating]
    ) -> npt.NDArray[np.integer]:
        indices = np.clip(
            np.around((received + self.order - 1) / 2), 0, self.order - 1
        ).astype(int)
        hard_bits = np.reshape(self.labeling[indices], shape=-1)
        return hard_bits

    @override
    def _demodulate_soft(
        self, received: npt.NDArray[np.floating | np.complexfloating], snr: float
    ) -> npt.NDArray[np.floating]:
        if self.order == 2:
            y = received / self.base_amplitude
            y = cast(npt.NDArray[np.floating], y)
            return self._demodulate_pam2_soft(y, snr / 1.0)
        elif self.order == 4 and self.labeling_parameter == "reflected":
            y = received / self.base_amplitude
            y = cast(npt.NDArray[np.floating], y)
            return vectorize(self._demodulate_pam4_soft_reflected)(y, snr / 5.0)  # type: ignore
        # Fall back to general implementation
        return super()._demodulate_soft(received, snr)

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
