import math

from ._constellations import constellation_qam
from ._labelings import labelings
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
    """

    def __init__(
        self, orders, base_amplitudes=1.0, phase_offset=0.0, labeling="reflected_2d"
    ):
        r"""
        Constructor for the class.

        Parameters:

            orders (Tuple(int, int) | int): A tuple $(M_\mathrm{I}, M_\mathrm{Q})$ with the orders of the in-phase and quadrature constellations, respectively; both $M_\mathrm{I}$ and $M_\mathrm{Q}$ must be powers of $2$. If specified as a single integer $M$, then it is assumed that $M_\mathrm{I} = M_\mathrm{Q} = \sqrt{M}$; in this case, $M$ must be an square power of $2$.

            base_amplitudes (Optional[Tuple(float, float) | float]): A tuple $(A_\mathrm{I}, A_\mathrm{Q})$ with the base amplitudes of the in-phase and quadrature constellations, respectively. If specified as a single float $A$, then it is assumed that $A_\mathrm{I} = A_\mathrm{Q} = A$. The default value is $1.0$.

            phase_offset (Optional[float]): The phase offset $\phi$ of the constellation. The default value is `0.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling of the modulation. Can be specified either as a 2D-array of integers (see [base class](/ref/Modulation) for details), or as a string. In the latter case, the string must be either `'natural_2d'` or `'reflected_2d'`. The default value is `'reflected_2d'`, corresponding to the Gray labeling.

        Examples:

            The square $16$-QAM modulation with $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ and $(A_\mathrm{I}, A_\mathrm{Q}) = (1, 1)$, and Gray labeling is depicted below.

            <figure markdown>
              ![16-QAM modulation with Gray labeling.](/figures/qam_16_gray.svg)
            </figure>

            >>> qam = komm.QAModulation(16)
            >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
                   -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
                   -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
                   -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
            >>> qam.labeling   #doctest: +NORMALIZE_WHITESPACE
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
            >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([-3.-2.j, -1.-2.j,  1.-2.j,  3.-2.j,
                   -3.+2.j, -1.+2.j,  1.+2.j,  3.+2.j])
            >>> qam.labeling  #doctest: +NORMALIZE_WHITESPACE
            array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
            >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1])
            array([-3.+2.j, -1.-2.j, -1.+2.j])
        """
        if isinstance(orders, (tuple, list)):
            order_I, order_Q = int(orders[0]), int(orders[1])
        else:
            order_I = order_Q = math.isqrt(orders)
        self._orders = (order_I, order_Q)

        if isinstance(base_amplitudes, (tuple, list)):
            base_amplitude_I, base_amplitude_Q = float(base_amplitudes[0]), float(
                base_amplitudes[1]
            )
        else:
            base_amplitude_I = base_amplitude_Q = float(base_amplitudes)
        self._base_amplitudes = (base_amplitude_I, base_amplitude_Q)

        self._phase_offset = float(phase_offset)

        allowed_labelings = ["natural_2d", "reflected_2d"]
        if labeling in allowed_labelings:
            labeling = labelings[labeling](self._orders)
        elif isinstance(labeling, str):
            raise ValueError(
                f"Only {allowed_labelings} or 2D-arrays are allowed for the labeling."
            )

        super().__init__(
            constellation_qam(self._orders, self._base_amplitudes, self._phase_offset),
            labeling,
        )

    def __repr__(self):
        args = "{}, base_amplitudes={}, phase_offset={}".format(
            self._orders, self._base_amplitudes, self._phase_offset
        )
        return "{}({})".format(self.__class__.__name__, args)
