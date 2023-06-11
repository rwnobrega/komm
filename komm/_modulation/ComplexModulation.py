import numpy as np

from .Modulation import Modulation


class ComplexModulation(Modulation):
    r"""
    General complex modulation scheme. A *complex modulation scheme* of order $M$ is defined by a *constellation* $\mathcal{S}$, which is an ordered subset (a list) of complex numbers, with $|\mathcal{S}| = M$, and a *binary labeling* $\mathcal{Q}$, which is a permutation of $[0: M)$. The order $M$ of the modulation must be a power of $2$.
    """

    def __init__(self, constellation, labeling):
        r"""
        Constructor for the class.

        Parameters:

            constellation (Array1D[complex]): The constellation $\mathcal{S}$ of the modulation. Must be a 1D-array containing $M$ complex numbers.

            labeling (Array1D[int]): The binary labeling $\mathcal{Q}$ of the modulation. Must be a 1D-array of integers corresponding to a permutation of $[0 : M)$.

        Examples:

            >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
            >>> mod.constellation
            array([ 0.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
            >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
            array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j, -1.+0.j])
        """
        super().__init__(np.array(constellation, dtype=complex), labeling)
