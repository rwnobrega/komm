import numpy as np

from .Modulation import Modulation
from .RealModulation import RealModulation


class PAModulation(RealModulation):
    r"""
    Pulse-amplitude modulation (PAM). It is a [real modulation scheme](/ref/RealModulation) in which the points of the constellation $\mathcal{S}$ are *uniformly arranged* in the real line. More precisely,
    $$
        \mathcal{S} = \\{ \pm (2i + 1)A : i \in [0 : M) \\},
    $$
    where $M$ is the *order* (a power of $2$), and $A$ is the *base amplitude*. The PAM constellation is depicted below for $M = 8$.

    |

    .. image:: figures/pam_8.svg
       :alt: 8-PAM constellation.
       :align: center
    """

    def __init__(self, order, base_amplitude=1.0, labeling="reflected"):
        r"""
        Constructor for the class.

        Parameters:

            order (int): The order $M$ of the modulation. It must be a power of $2$.

            base_amplitude (Optional[float]): The base amplitude $A$ of the constellation. The default value is `1.0`.

            labeling (Optional[Array1D[int] | str]): The binary labeling $\mathcal{Q}$ of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of $[0 : M)$, or as a string, in which case must be one of `'natural'` or `'reflected'`. The default value is `'reflected'` (Gray code).

        Examples:

            >>> pam = komm.PAModulation(4, base_amplitude=2.0)
            >>> pam.constellation
            array([-6., -2.,  2.,  6.])
            >>> pam.labeling
            array([0, 1, 3, 2])
            >>> pam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
            array([-6.,  2., -6., -2., -2.])
        """
        constellation = base_amplitude * np.arange(-order + 1, order, step=2, dtype=int)

        if isinstance(labeling, str):
            if labeling in ["natural", "reflected"]:
                labeling = getattr(Modulation, "_labeling_" + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._base_amplitude = float(base_amplitude)

    def __repr__(self):
        args = "{}, base_amplitude={}".format(self._order, self._base_amplitude)
        return "{}({})".format(self.__class__.__name__, args)
