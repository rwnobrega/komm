import numpy as np


class TransmitFilter:
    """
    Transmit filter.
    """
    def __init__(self, pulse, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameters:

        :code:`pulse` : :class:`komm.Pulse`
            The pulse filter.

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples (of the output) per symbol (of the input).
        """
        self.Pulse = pulse
        self._samples_per_symbol = int(samples_per_symbol)

    @property
    def pulse(self):
        """
        The pulse filter. This property is read-only.
        """
        return self.Pulse

    @property
    def samples_per_symbol(self):
        """
        The number of samples per symbol. This property is read-only.
        """
        return self._samples_per_symbol

    def __call__(self, inp):
        """
        Formats a sequence of symbols.

        .. rubric:: Input

        :code:`inp` : 1D-array of :obj:`float`
            The input signal, containing symbols of a modulation.

        .. rubric:: Output

        :code:`outp` : 1D-array of :obj:`float`
            The output signal, formatted.
        """
        sps = self._samples_per_symbol
        t0, t1 = self.Pulse.interval
        t = np.arange(t0, t1, step=1/sps)
        taps = (np.vectorize(self.Pulse.impulse_response))(t)
        inp_interp = np.zeros((len(inp) - 1) * sps + 1, dtype=float)
        inp_interp[::sps] = inp
        outp = np.convolve(taps, inp_interp)
        return outp
