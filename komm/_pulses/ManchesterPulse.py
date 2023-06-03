from .Pulse import Pulse


class ManchesterPulse(Pulse):
    r"""
    Manchester pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) =
        \begin{cases}
            -1, & 0 \leq t <  1/2, \\
            1, & 1/2 \leq t < 1, \\
            0, & \text{otherwise},
        \end{cases}

    The Manchester pulse is depicted below.

    .. image:: figures/pulse_manchester.png
       :alt: Manchester pulse
       :align: center
    """

    def __init__(self):
        r"""
        Constructor for the class. It expects no parameters.

        .. rubric:: Examples

        >>> pulse = komm.ManchesterPulse()
        """

        def impulse_response(t):
            return -1.0 * (0 <= t < 0.5) + 1.0 * (0.5 <= t < 1)

        super().__init__(impulse_response, interval=(0.0, 1.0))

    def __repr__(self):
        args = ""
        return "{}({})".format(self.__class__.__name__, args)
