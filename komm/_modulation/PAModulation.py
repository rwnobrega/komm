import numpy as np

from .Modulation import Modulation
from .RealModulation import RealModulation


class PAModulation(RealModulation):
    """
    Pulse-amplitude modulation (PAM). It is a real modulation scheme (:class:`RealModulation`) in which the points of the constellation :math:`\\mathcal{S}` are *uniformly arranged* in the real line. More precisely,

    .. math::
        \\mathcal{S} = \\{ \\pm (2i + 1)A : i \\in [0 : M) \\},

    where :math:`M` is the *order* (a power of :math:`2`), and :math:`A` is the *base amplitude*. The PAM constellation is depicted below for :math:`M = 8`.

    |

    .. image:: figures/pam_8.png
       :alt: 8-PAM constellation.
       :align: center
    """
    def __init__(self, order, base_amplitude=1.0, labeling='reflected'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`order` : :obj:`int`
            The order :math:`M` of the modulation. It must be a power of :math:`2`.

        :code:`base_amplitude` : :obj:`float`, optional
            The base amplitude :math:`A` of the constellation. The default value is :code:`1.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected'`. The default value is :code:`'reflected'` (Gray code).

        .. rubric:: Examples

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
            if labeling in ['natural', 'reflected']:
                labeling = getattr(Modulation, '_labeling_' + labeling)(order)
            else:
                raise ValueError("Only 'natural' or 'reflected' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._base_amplitude = float(base_amplitude)

    def __repr__(self):
        args = '{}, base_amplitude={}'.format(self._order, self._base_amplitude)
        return '{}({})'.format(self.__class__.__name__, args)
