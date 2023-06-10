import numpy as np


class AWGNChannel:
    r"""
    Additive white Gaussian noise (AWGN) channel. It is defined by
    $$
        Y_n = X_n + Z_n,
    $$
    where $X_n$ is the channel *input signal*, $Y_n$ is the channel *output signal*, and $Z_n$ is the *noise*, which is :term:`i.i.d.` according to a Gaussian distribution with zero mean. The channel *signal-to-noise ratio* is calculated by
    $$
        \mathrm{SNR} = \frac{P}{N},
    $$
    where $P = \mathrm{E}[X^2_n]$ is the average power of the input signal, and $N = \mathrm{E}[Z^2_n]$ is the average power (and variance) of the noise. See :cite:`Cover.Thomas.06` (Ch. 9).

    To invoke the channel, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, snr=np.inf, signal_power=1.0):
        r"""Constructor for the class.

        Parameters:

            snr (:obj:`float`, optional): The channel signal-to-noise ratio $\mathrm{SNR}$ (linear, not decibel). The default value is `np.inf`, which corresponds to a noiseless channel.

            signal_power (:obj:`float` or :obj:`str`, optional): The input signal power $P$. If equal to the string `'measured'`, then every time the channel is invoked the input signal power will be computed from the input itself (i.e., its squared Euclidean norm). The default value is `1.0`.

        Examples:

            >>> awgn = komm.AWGNChannel(snr=100.0, signal_power=5.0)
            >>> x = [1.0, 3.0, -3.0, -1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 3.0]
            >>> y = awgn(x); y  #doctest: +SKIP
            array([ 0.91623839,  2.66229342, -2.96852259, -1.07689368, -0.89296933,
                    0.80128101,  3.34942297,  1.24031682, -0.84460601,  2.96762221])
        """
        self.snr = snr
        self.signal_power = signal_power

    @property
    def snr(self):
        r"""
        The signal-to-noise ratio $\mathrm{SNR}$ (linear, not decibel) of the channel. This is a read-and-write property.
        """
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = float(value)

    @property
    def signal_power(self):
        r"""
        The input signal power $P$. This is a read-and-write property.
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
        Returns the channel capacity $C$. It is given by $C = \frac{1}{2}\log_2(1 + \mathrm{SNR})$, in bits per dimension.

        Examples:

            >>> awgn = komm.AWGNChannel(snr=63.0)
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
