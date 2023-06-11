import numpy as np

from .ComplexModulation import ComplexModulation
from .Modulation import Modulation


class QAModulation(ComplexModulation):
    r"""
    Quadrature-amplitude modulation (QAM). It is a [complex modulation scheme](/ref/ComplexModulation) in which the constellation $\mathcal{S}$ is given as a Cartesian product of two [PAM](/ref/PAModulation) constellations, namely, the *in-phase constellation*, and the *quadrature constellation*. More precisely,
    $$
        \mathcal{S} = \\{ [\pm(2i_\mathrm{I} + 1)A_\mathrm{I} \pm \mathrm{j}(2i_\mathrm{Q} + 1)A_\mathrm{Q}] \exp(\mathrm{j}\phi) : i_\mathrm{I} \in [0 : M_\mathrm{I}), i_\mathrm{Q} \in [0 : M_\mathrm{Q}) \\},
    $$
    where $M_\mathrm{I}$ and $M_\mathrm{Q}$ are the *orders* (powers of $2$), and $A_\mathrm{I}$ and $A_\mathrm{Q}$ are the *base amplitudes* of the in-phase and quadrature constellations, respectively. Also, $\phi$ is the *phase offset*. The size of the resulting complex-valued constellation is $M = M_\mathrm{I} M_\mathrm{Q}$, a power of $2$. The QAM constellation is depicted below for $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 4)$ with $A_\mathrm{I} = A_\mathrm{Q} = A$, and for $(M_\mathrm{I}, M_\mathrm{Q}) = (4, 2)$ with $A_\mathrm{I} = A$ and $A_\mathrm{Q} = 2A$; in both cases, $\phi = 0$.

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/qam_16.svg
       :alt: 16-QAM constellation

    .. |fig2| image:: figures/qam_8.svg
       :alt: 8-QAM constellation

    .. |quad| unicode:: 0x2001
       :trim:
    """

    def __init__(self, orders, base_amplitudes=1.0, phase_offset=0.0, labeling="reflected_2d"):
        r"""
        Constructor for the class.

        Parameters:

            orders (:obj:`(int, int)` or :obj:`int`): A tuple $(M_\mathrm{I}, M_\mathrm{Q})$ with the orders of the in-phase and quadrature constellations, respectively; both $M_\mathrm{I}$ and $M_\mathrm{Q}$ must be powers of $2$. If specified as a single integer $M$, then it is assumed that $M_\mathrm{I} = M_\mathrm{Q} = \sqrt{M}$; in this case, $M$ must be an square power of $2$.

            base_amplitudes (:obj:`(float, float)` or :obj:`float`, optional): A tuple $(A_\mathrm{I}, A_\mathrm{Q})$ with the base amplitudes of the in-phase and quadrature constellations, respectively.  If specified as a single float $A$, then it is assumed that $A_\mathrm{I} = A_\mathrm{Q} = A$. The default value is $1.0$.

            phase_offset (:obj:`float`, optional): The phase offset $\phi$ of the constellation. The default value is `0.0`.

            labeling ((1D-array of :obj:`int`) or :obj:`str`, optional): The binary labeling $\mathcal{Q}$ of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of $[0 : M)$, or as a string, in which case must be one of `'natural'` or `'reflected_2d'`. The default value is `'reflected_2d'` (Gray code).

        Examples:

            >>> qam = komm.QAModulation(16)
            >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
                   -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
                   -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
                   -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
            >>> qam.labeling
            array([ 0,  1,  3,  2,  4,  5,  7,  6, 12, 13, 15, 14,  8,  9, 11, 10])
            >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0])
            array([-3.+1.j, -3.-1.j])

            >>> qam = komm.QAModulation(orders=(4, 2), base_amplitudes=(1.0, 2.0))
            >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
            array([-3.-2.j, -1.-2.j,  1.-2.j,  3.-2.j, -3.+2.j, -1.+2.j,  1.+2.j,  3.+2.j])
            >>> qam.labeling
            array([0, 1, 3, 2, 4, 5, 7, 6])
            >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1])
            array([-3.+2.j, -1.-2.j, -1.+2.j])
        """
        if isinstance(orders, (tuple, list)):
            order_I, order_Q = int(orders[0]), int(orders[1])
            self._orders = (order_I, order_Q)
        else:
            order_I = order_Q = int(np.sqrt(orders))
            self._orders = int(orders)

        if isinstance(base_amplitudes, (tuple, list)):
            base_amplitude_I, base_amplitude_Q = float(base_amplitudes[0]), float(base_amplitudes[1])
            self._base_amplitudes = (base_amplitude_I, base_amplitude_Q)
        else:
            base_amplitude_I = base_amplitude_Q = float(base_amplitudes)
            self._base_amplitudes = base_amplitude_I

        constellation_I = base_amplitude_I * np.arange(-order_I + 1, order_I, step=2, dtype=int)
        constellation_Q = base_amplitude_Q * np.arange(-order_Q + 1, order_Q, step=2, dtype=int)
        constellation = (constellation_I + 1j * constellation_Q[np.newaxis].T).flatten() * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling == "natural":
                labeling = Modulation._labeling_natural(order_I * order_Q)
            elif labeling == "reflected_2d":
                labeling = Modulation._labeling_reflected_2d(order_I, order_Q)
            else:
                raise ValueError(
                    "Only 'natural' or 'reflected_2d' are supported for {}".format(self.__class__.__name__)
                )

        super().__init__(constellation, labeling)

        self._orders = orders
        self._base_amplitudes = base_amplitudes
        self._phase_offset = float(phase_offset)

    def __repr__(self):
        args = "{}, base_amplitudes={}, phase_offset={}".format(self._orders, self._base_amplitudes, self._phase_offset)
        return "{}({})".format(self.__class__.__name__, args)
