import numpy as np

from .FormattingPulse import FormattingPulse


class GaussianPulse(FormattingPulse):
    r"""
    Gaussian pulse. It is a formatting pulse with impulse response given by
    $$
        h(t) = \mathrm{e}^{-\frac{1}{2} (2 \pi \bar{B} t)^2}
    $$
    where the $\bar{B} = B / \sqrt{\ln 2}$, and $B$ is the half-power bandwidth of the filter.

    The Gaussian pulse is depicted below for $B = 0.5$, and for $B = 1$.

    <div class="centered" markdown>
      <span>
        ![Gaussian pulse with half-power bandwidth of 0.5.](/figures/pulse_gaussian_50.svg)
      </span>
      <span>
        ![Gaussian pulse with half-power bandwidth of 1.](/figures/pulse_gaussian_100.svg)
      </span>
    </div>
    """

    def __init__(self, half_power_bandwidth, length_in_symbols):
        r"""
        Constructor for the class.

        Parameters:
            half_power_bandwidth (float): The half-power bandwidth $B$ of the pulse.

            length_in_symbols (int): The length (span) of the truncated impulse response, in symbols.

        Examples:
            >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.5, length_in_symbols=4)

            >>> pulse = komm.GaussianPulse(half_power_bandwidth=1.0, length_in_symbols=2)
        """
        B = self._half_power_bandwidth = float(half_power_bandwidth)
        L = self._length_in_symbols = int(length_in_symbols)
        B_bar = B / np.sqrt(np.log(2))

        def impulse_response(t):
            return np.exp(-0.5 * (2 * np.pi * B_bar * t) ** 2)

        def frequency_response(f):
            return 1 / (np.sqrt(2 * np.pi) * B_bar) * np.exp(-0.5 * (f / B_bar) ** 2)

        super().__init__(impulse_response, frequency_response, interval=(-L / 2, L / 2))

    def __repr__(self):
        args = "half_power_bandwidth={}, length_in_symbols={}".format(
            self._half_power_bandwidth, self._length_in_symbols
        )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def half_power_bandwidth(self):
        r"""
        The half-power bandwidth $B$ of the pulse.
        """
        return self._half_power_bandwidth

    @property
    def length_in_symbols(self):
        r"""
        The length (span) of the truncated impulse response.
        """
        return self._length_in_symbols
