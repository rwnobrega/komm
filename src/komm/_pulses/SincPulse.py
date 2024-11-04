import numpy as np

from .FormattingPulse import FormattingPulse


class SincPulse(FormattingPulse):
    r"""
    Sinc pulse. It is a formatting pulse with impulse response given by
    $$
        h(t) = \operatorname{sinc}(t) = \frac{\sin(\pi t)}{\pi t}.
    $$
    The sinc pulse is depicted below.

    <figure markdown>
      ![Sinc pulse.](/figures/pulse_sinc.svg)
    </figure>
    """

    def __init__(self, length_in_symbols):
        r"""
        Constructor for the class.

        Parameters:
            length_in_symbols (int): The length (span) of the truncated impulse response, in symbols.

        Examples:
            >>> pulse = komm.SincPulse(length_in_symbols=64)
        """
        L = self._length_in_symbols = int(length_in_symbols)

        def impulse_response(t):
            return np.sinc(t)

        def frequency_response(f):
            return 1.0 * (abs(f) < 0.5)

        super().__init__(impulse_response, frequency_response, interval=(-L / 2, L / 2))

    @property
    def length_in_symbols(self):
        r"""
        The length (span) of the truncated impulse response.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = "length_in_symbols={}".format(self._length_in_symbols)
        return "{}({})".format(self.__class__.__name__, args)
