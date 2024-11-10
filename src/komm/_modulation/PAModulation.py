import numpy as np

from ._constellations import constellation_pam
from ._labelings import labelings
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
    """

    def __init__(self, order, base_amplitude=1.0, labeling="reflected"):
        r"""
        Constructor for the class.

        Parameters:
            order (int): The order $M$ of the modulation. It must be a power of $2$.

            base_amplitude (Optional[float]): The base amplitude $A$ of the constellation. The default value is `1.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

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
        self._constructor_kwargs = {
            "order": order,
            "base_amplitude": base_amplitude,
            "labeling": labeling,
        }

        self._base_amplitude = float(base_amplitude)

        allowed_labelings = ["natural", "reflected"]
        if labeling in allowed_labelings:
            labeling = labelings[labeling](order)
        elif isinstance(labeling, str):
            raise ValueError(
                f"only {allowed_labelings} or 2D-arrays are allowed for the labeling"
            )

        super().__init__(
            constellation=constellation_pam(order, self._base_amplitude),
            labeling=labeling,
        )

    def __repr__(self):
        order, base_amplitude, labeling = self._constructor_kwargs.values()
        args = f"{order}, base_amplitude={base_amplitude}, labeling='{labeling}'"
        return f"{self.__class__.__name__}({args})"

    def _demodulate_hard(self, received):
        indices = np.clip(
            np.around((received + self._order - 1) / 2),
            0,
            self._order - 1,
        ).astype(int)
        return np.reshape(self._labeling[indices], shape=-1)

    @staticmethod
    def _demodulate_soft_pam2(y, gamma):
        # SA15, eq. (3.65).
        return -4.0 * gamma * y

    @staticmethod
    def _demodulate_soft_pam4_reflected(y, gamma):
        soft_bits = np.empty(2 * y.size, dtype=float)
        # For bit_0: SA15, eq. (5.15)
        soft_bits[0::2] = -8.0 * gamma + np.log(
            np.cosh(6.0 * gamma * y) / np.cosh(2.0 * gamma * y),
        )
        # For bit_1: SA15, eq. (5.6)
        soft_bits[1::2] = -8.0 * gamma * y + np.log(
            np.cosh(2.0 * gamma * (y + 2.0)) / np.cosh(2.0 * gamma * (y - 2.0)),
        )
        return soft_bits

    def _demodulate_soft(self, received, channel_snr=1.0):
        if self._order == 2 and self._constructor_kwargs["labeling"] == "reflected":
            return self._demodulate_soft_pam2(
                y=received / self._base_amplitude,
                gamma=channel_snr,
            )
        elif self._order == 4 and self._constructor_kwargs["labeling"] == "reflected":
            return self._demodulate_soft_pam4_reflected(
                y=received / self._base_amplitude,
                gamma=channel_snr / 5.0,
            )
        return super()._demodulate_soft(received, channel_snr)
