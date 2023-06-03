import numpy as np

from .ComplexModulation import ComplexModulation
from .Modulation import Modulation


class PSKModulation(ComplexModulation):
    r"""
    Phase-shift keying (PSK) modulation. It is a complex modulation scheme (:class:`ComplexModulation`) in which the points of the constellation :math:`\mathcal{S}` are *uniformly arranged* in a circle. More precisely,

    .. math::
        \mathcal{S} = \left \{ A \exp \left( \mathrm{j} \frac{2 \pi i}{M} \right) \exp(\mathrm{j} \phi) : i \in [0 : M) \right \}

    where :math:`M` is the *order* (a power of :math:`2`), :math:`A` is the *amplitude*, and :math:`\phi` is the *phase offset* of the modulation. The PSK constellation is depicted below for :math:`M = 8`.

    .. image:: figures/psk_8.png
       :alt: 8-PSK constellation.
       :align: center
    """

    def __init__(self, order, amplitude=1.0, phase_offset=0.0, labeling="reflected"):
        r"""
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`amplitude` : :obj:`float`, optional
            The amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).

        .. rubric:: Examples

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
