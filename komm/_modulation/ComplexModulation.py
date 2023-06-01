import numpy as np
from .Modulation import Modulation

class ComplexModulation(Modulation):
    """
    General complex modulation scheme. A *complex modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of complex numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.
    """
    def __init__(self, constellation, labeling):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`complex`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array containing :math:`M` complex numbers.

        :code:`labeling` : 1D-array of :obj:`int`
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Must be a 1D-array of integers corresponding to a permutation of :math:`[0 : M)`.

        .. rubric:: Examples

        >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
        >>> mod.constellation
        array([ 0.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
        >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j, -1.+0.j])
        """
        super().__init__(np.array(constellation, dtype=complex), labeling)
