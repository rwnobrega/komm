import numpy as np

from .Modulation import Modulation


class RealModulation(Modulation):
    r"""
    General real modulation scheme. A *real modulation scheme* of order $M$ is defined by a *constellation* $\mathcal{S}$, which is an ordered subset (a list) of real numbers, with $|\mathcal{S}| = M$, and a *binary labeling* $\mathcal{Q}$, which is a permutation of $[0: M)$. The order $M$ of the modulation must be a power of $2$.
    """

    def __init__(self, constellation, labeling):
        r"""
        Constructor for the class.

        Parameters:

            constellation (1D-array of :obj:`complex`): The constellation $\mathcal{S}$ of the modulation. Must be a 1D-array containing $M$ real numbers.

            labeling (1D-array of :obj:`int`): The binary labeling $\mathcal{Q}$ of the modulation. Must be a 1D-array of integers corresponding to a permutation of $[0 : M)$.

        Examples:

            >>> mod = komm.RealModulation(constellation=[-0.5, 0, 0.5, 2], labeling=[0, 1, 3, 2])
            >>> mod.constellation
            array([-0.5,  0. ,  0.5,  2. ])
            >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
            array([-0.5,  0.5, -0.5,  0. ,  0. ])
        """
        super().__init__(np.array(constellation, dtype=float), labeling)
