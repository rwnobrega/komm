from .Pulse import Pulse


class RectangularPulse(Pulse):
    """
    Rectangular pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) =
        \\begin{cases}
            1, & 0 \\leq t < w, \\\\
            0, & \\text{otherwise}.
        \\end{cases},

    where :math:`w` is the *width* of the pulse, which must satisfy :math:`0 \\leq w \\leq 1`. The rectangular pulse is depicted below for :math:`w = 1` (called the :term:`NRZ` pulse), and for :math:`w = 0.5` (called the halfway :term:`RZ` pulse).

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/pulse_rectangular_nrz.png
       :alt: Rectangular NRZ pulse

    .. |fig2| image:: figures/pulse_rectangular_rz.png
       :alt: Rectangular RZ pulse

    .. |quad| unicode:: 0x2001
       :trim:
    """

    def __init__(self, width=1.0):
        """
        Constructor for the class. It expects the following parameter:

        :code:`width` : :obj:`float`
            The width :math:`w` of the pulse. Must satisfy :math:`0 \\leq w \\leq 1`. The default value is :code:`1.0`.

        .. rubric:: Examples

        >>> pulse = komm.RectangularPulse(width=1.0)

        >>> pulse = komm.RectangularPulse(width=0.5)
        """
        w = self._width = float(width)

        def impulse_response(t):
            return 1.0 * (0 <= t < w)

        super().__init__(impulse_response, interval=(0.0, 1.0))

    def __repr__(self):
        args = "width={}".format(self._width)
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def width(self):
        """
        The width :math:`w` of the pulse. This property is read-only.
        """
        return self._width
