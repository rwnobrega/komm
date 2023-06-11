import numpy as np


class TransmitFilter:
    r"""
    Transmit filter.
    """

    def __init__(self, pulse, samples_per_symbol):
        r"""
        Constructor for the class.

        Parameters:

            pulse (FormattingPulse): The pulse filter.

            samples_per_symbol (int): The number of samples (of the output) per symbol (of the input).
        """
        self.Pulse = pulse
        self._samples_per_symbol = int(samples_per_symbol)

    @property
    def pulse(self):
        r"""
        The pulse filter.
        """
        return self.Pulse

    @property
    def samples_per_symbol(self):
        r"""
        The number of samples per symbol.
        """
        return self._samples_per_symbol

    def __call__(self, inp):
        r"""
        Formats a sequence of symbols.

        Parameters:

            inp (Array1D[float]): The input signal, containing symbols of a modulation.

        Returns:

            outp (Array1D[float]): The output signal, formatted.
        """
        sps = self._samples_per_symbol
        t0, t1 = self.Pulse.interval
        t = np.arange(t0, t1, step=1 / sps)
        taps = (np.vectorize(self.Pulse.impulse_response))(t)
        inp_interp = np.zeros((len(inp) - 1) * sps + 1, dtype=float)
        inp_interp[::sps] = inp
        outp = np.convolve(taps, inp_interp)
        return outp
