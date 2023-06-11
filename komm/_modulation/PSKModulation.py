import numpy as np

from .ComplexModulation import ComplexModulation
from .Modulation import Modulation


class PSKModulation(ComplexModulation):
    r"""
    Phase-shift keying (PSK) modulation. It is a [complex modulation scheme](/ref/ComplexModulation) in which the points of the constellation $\mathcal{S}$ are *uniformly arranged* in a circle. More precisely,
    $$
        \mathcal{S} = \left \\{ A \exp \left( \mathrm{j} \frac{2 \pi i}{M} \right) \exp(\mathrm{j} \phi) : i \in [0 : M) \right \\}
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *amplitude*, and $\phi$ is the *phase offset* of the modulation. The PSK constellation is depicted below for $M = 8$.

    .. image:: figures/psk_8.svg
       :alt: 8-PSK constellation.
       :align: center
    """

    def __init__(self, order, amplitude=1.0, phase_offset=0.0, labeling="reflected"):
        r"""
        Constructor for the class.

        Parameters:

            order (int): The order $M$ of the modulation. It must be a power of $2$.

            amplitude (Optional[float]): The amplitude $A$ of the constellation. The default value is `1.0`.

            phase_offset (Optional[float]): The phase offset $\phi$ of the constellation. The default value is `0.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling $\mathcal{Q}$ of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of $[0 : M)$, or as a string, in which case must be one of `'natural'` or `'reflected'`. The default value is `'reflected'` (Gray code).

        Examples:

            >>> psk = komm.PSKModulation(4, phase_offset=np.pi/4)
            >>> psk.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([ 0.70710678+0.70710678j, -0.70710678+0.70710678j, -0.70710678-0.70710678j,  0.70710678-0.70710678j])
            >>> psk.labeling
            array([0, 1, 3, 2])
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
