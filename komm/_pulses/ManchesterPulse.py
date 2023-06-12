from .FormattingPulse import FormattingPulse


class ManchesterPulse(FormattingPulse):
    r"""
    Manchester pulse. It is a formatting pulse with impulse response given by
    $$
        h(t) =
        \begin{cases}
            -1, & 0 \leq t <  1/2, \\\\
            1, & 1/2 \leq t < 1, \\\\
            0, & \text{otherwise}.
        \end{cases}
    $$
    The Manchester pulse is depicted below.

    <figure markdown>
      ![Manchester pulse](/figures/pulse_manchester.svg)
    </figure>
    """

    def __init__(self):
        r"""
        Constructor for the class. It expects no parameters.

        Examples:

            >>> pulse = komm.ManchesterPulse()
        """

        def impulse_response(t):
            return -1.0 * (0 <= t < 0.5) + 1.0 * (0.5 <= t < 1)

        super().__init__(impulse_response, interval=(0.0, 1.0))

    def __repr__(self):
        args = ""
        return "{}({})".format(self.__class__.__name__, args)
