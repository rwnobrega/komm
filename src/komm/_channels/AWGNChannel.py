import numpy as np


class AWGNChannel:
    r"""
    Additive white Gaussian noise (AWGN) channel. It is defined by
    $$
        Y_n = X_n + Z_n,
    $$
    where $X_n$ is the channel *input signal*, $Y_n$ is the channel *output signal*, and $Z_n$ is the *noise*, which is iid according to a Gaussian distribution with zero mean. The channel *signal-to-noise ratio* is calculated by
    $$
        \snr = \frac{P}{N},
    $$
    where $P = \mathrm{E}[X^2_n]$ is the average power of the input signal, and $N = \mathrm{E}[Z^2_n]$ is the average power (and variance) of the noise. For more details, see <cite>CT06, Ch. 9</cite>.

    To invoke the channel, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, signal_power, snr=np.inf):
        r"""Constructor for the class.

        Parameters:

            signal_power (float | str): The input signal power $P$. If equal to the string `'measured'`, then every time the channel is invoked the input signal power will be computed from the input itself (i.e., its squared Euclidean norm).

            snr (Optional[float]): The channel signal-to-noise ratio $\snr$ (linear, not decibel). The default value is `np.inf`, which corresponds to a noiseless channel.

        Examples:

            >>> np.random.seed(1)
            >>> awgn = komm.AWGNChannel(snr=200.0, signal_power=5.0)
            >>> x = [1.0, 3.0, -3.0, -1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 3.0]
            >>> y = awgn(x); np.around(y, decimals=2)  # doctest: +NORMALIZE_WHITESPACE
            array([ 1.26,  2.9 , -3.08, -1.17, -0.86,  0.64,  3.28,  0.88, -0.95,  2.96])
        """
        self.snr = snr
        self.signal_power = signal_power

    @property
    def snr(self):
        r"""
        The signal-to-noise ratio $\snr$ (linear, not decibel) of the channel.
        """
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = float(value)

    @property
    def signal_power(self):
        r"""
        The input signal power $P$.
        """
        return self._signal_power

    @signal_power.setter
    def signal_power(self, value):
        if value == "measured":
            self._signal_power = value
        else:
            self._signal_power = float(value)

    def capacity(self):
        r"""
        Returns the channel capacity $C$. It is given by $C = \frac{1}{2}\log_2(1 + \snr)$, in bits per dimension.

        Examples:

            >>> awgn = komm.AWGNChannel(signal_power=1.0, snr=63.0)
            >>> awgn.capacity()
            3.0
        """
        return 0.5 * np.log1p(self._snr) / np.log(2.0)

    def __call__(self, input_signal):
        input_signal = np.array(input_signal)
        size = input_signal.size

        if self._signal_power == "measured":
            signal_power = np.linalg.norm(input_signal) ** 2 / size
        else:
            signal_power = self._signal_power

        noise_power = signal_power / self._snr

        if input_signal.dtype == complex:
            noise = np.sqrt(noise_power / 2) * (np.random.normal(size=size) + 1j * np.random.normal(size=size))
        else:
            noise = np.sqrt(noise_power) * np.random.normal(size=size)

        return input_signal + noise

    def __repr__(self):
        args = "snr={}, signal_power={}".format(self._snr, self._signal_power)
        return "{}({})".format(self.__class__.__name__, args)
