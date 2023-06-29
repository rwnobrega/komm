import numpy as np

from .Modulation import Modulation


class ASKModulation(Modulation):
    r"""
    Amplitude-shift keying (ASK) modulation. It is a complex [modulation scheme](/ref/Modulation) in which the points of the constellation symbols are *uniformly arranged* in a ray. More precisely, the rows of the constellation $\mathbf{X}$ are given by
    $$
        x_i = iA \exp(\mathrm{j}\phi), \quad i \in [0 : M),
    $$
    where $M$ is the *order* (a power of $2$), $A$ is the *base amplitude*, and $\phi$ is the *phase offset* of the modulation. The ASK constellation is depicted below for $M = 4$.

    <figure markdown>
      ![4-ASK constellation.](/figures/ask_4.svg)
    </figure>
    """

    def __init__(self, order, base_amplitude=1.0, phase_offset=0.0, labeling="reflected"):
        r"""
        Constructor for the class.

        Parameters:

            order (int): The order $M$ of the modulation. It must be a power of $2$.

            base_amplitude (Optional[float]): The base amplitude $A$ of the constellation. The default value is `1.0`.

            phase_offset (Optional[float]): The phase offset $\phi$ of the constellation. The default value is `0.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural'` or `'reflected'`. The default value is `'reflected'`, corresponding to the Gray labeling.

        Examples:

            The ASK constellation with order $M = 4$, base amplitude $A = 1$, and Gray labeling is depicted below.

            <figure markdown>
              ![4-ASK constellation with Gray labeling.](/figures/ask_4_gray.svg)
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
        self._constructor_kwargs = {
            "order": order,
            "base_amplitude": base_amplitude,
            "phase_offset": phase_offset,
            "labeling": labeling,
        }

        constellation = base_amplitude * np.arange(order, dtype=int) * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling in ["natural", "reflected"]:
                labeling = getattr(Modulation, "_labeling_" + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._base_amplitude = float(base_amplitude)
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        order, base_amplitude, phase_offset, labeling = self._constructor_kwargs.values()
        args = f"{order}, base_amplitude={base_amplitude}, phase_offset={phase_offset}, labeling={labeling}"
        return f"{self.__class__.__name__}({args})"
