import numpy as np

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
    """

    def __init__(self, order, amplitude=1.0, phase_offset=0.0, labeling="reflected"):
        r"""
        Constructor for the class.

        Parameters:

            order (int): The order $M$ of the modulation. It must be a power of $2$.

            amplitude (Optional[float]): The amplitude $A$ of the constellation. The default value is `1.0`.

            phase_offset (Optional[float]): The phase offset $\phi$ of the constellation. The default value is `0.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

        The PSK modulation with order $M = 4$, base amplitude $A = 1$, phase offset $\phi = \pi/4$, and Gray labeling is depicted below.

        <figure markdown>
          ![4-PSK modulation with Gray labeling.](/figures/psk_4_gray.svg)
        </figure>

        Examples:

            >>> psk = komm.PSKModulation(4, phase_offset=np.pi/4.0)
            >>> psk.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([ 0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678-0.70710678j])
            >>> psk.labeling
            array([[0, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1]])
            >>> psk.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])  #doctest: +NORMALIZE_WHITESPACE
            array([ 0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678+0.70710678j])
        """
        constellation = amplitude * np.exp(2j * np.pi * np.arange(order) / order) * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling in ["natural", "reflected"]:
                labeling = getattr(Modulation, "_labeling_" + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._amplitude = float(amplitude)
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        args = "{}, amplitude={}, phase_offset={}".format(self._order, self._amplitude, self._phase_offset)
        return "{}({})".format(self.__class__.__name__, args)
