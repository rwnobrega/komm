import numpy as np

from .ComplexModulation import ComplexModulation
from .Modulation import Modulation


class ASKModulation(ComplexModulation):
    """
    Amplitude-shift keying (ASK) modulation. It is a complex modulation scheme (:class:`ComplexModulation`) in which the points of the constellation :math:`\\mathcal{S}` are *uniformly arranged* in a ray. More precisely,

    .. math::

        \\mathcal{S} = \\{ iA \\exp(\\mathrm{j}\\phi): i \\in [0 : M) \\},

    where :math:`M` is the *order* (a power of :math:`2`), :math:`A` is the *base amplitude*, and :math:`\\phi` is the *phase offset* of the modulation.  The ASK constellation is depicted below for :math:`M = 4`.

    .. image:: figures/ask_4.png
       :alt: 4-ASK constellation.
       :align: center
    """

    def __init__(self, order, base_amplitude=1.0, phase_offset=0.0, labeling="reflected"):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).

        .. rubric:: Examples

        >>> ask = komm.ASKModulation(4, base_amplitude=2.0)
        >>> ask.constellation
        array([0.+0.j, 2.+0.j, 4.+0.j, 6.+0.j])
        >>> ask.labeling
        array([0, 1, 3, 2])
        >>> ask.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        array([0.+0.j, 4.+0.j, 0.+0.j, 2.+0.j, 2.+0.j])
        >>> ask.demodulate([(0.99+0.3j), (1.01-0.5j), (4.99+0.7j), (5.01-0.9j)])
        array([0, 0, 1, 0, 1, 1, 0, 1])
        """
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
        args = "{}, base_amplitude={}, phase_offset={}".format(self._order, self._base_amplitude, self._phase_offset)
        return "{}({})".format(self.__class__.__name__, args)
