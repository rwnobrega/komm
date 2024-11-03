from typing import Literal

import numpy as np
from attrs import frozen


@frozen
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

    To invoke the channel, call the object giving the input signal as parameter (see example below).

    Attributes:
        signal_power (float | str): The input signal power $P$. If equal to the string `'measured'`, then every time the channel is invoked the input signal power will be computed from the input itself (i.e., its squared Euclidean norm).

        snr (Optional[float]): The channel signal-to-noise ratio $\snr$ (linear, not decibel). The default value is `np.inf`, which corresponds to a noiseless channel.

    Examples:
        >>> np.random.seed(1)
        >>> awgn = komm.AWGNChannel(signal_power=5.0, snr=200.0)
        >>> x = [1.0, 3.0, -3.0, -1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 3.0]
        >>> y = awgn(x); np.around(y, decimals=2)  # doctest: +NORMALIZE_WHITESPACE
        array([ 1.26,  2.9 , -3.08, -1.17, -0.86,  0.64,  3.28,  0.88, -0.95,  2.96])
    """

    signal_power: float | Literal["measured"]
    snr: float = np.inf

    @property
    def noise_power(self):
        r"""
        The noise power $N$.
        """
        if self.signal_power == "measured":
            raise ValueError(
                "The noise power cannot be calculated when the signal power is measured."
            )
        return self.signal_power / self.snr

    def capacity(self):
        r"""
        Returns the channel capacity $C$. It is given by $C = \frac{1}{2}\log_2(1 + \snr)$, in bits per dimension.

        Examples:
            >>> awgn = komm.AWGNChannel(signal_power=1.0, snr=63.0)
            >>> awgn.capacity()
            np.float64(3.0)
        """
        return 0.5 * np.log1p(self.snr) / np.log(2.0)

    def __call__(self, input_signal):
        input_signal = np.array(input_signal)
        size = input_signal.size

        if self.signal_power == "measured":
            signal_power = np.linalg.norm(input_signal) ** 2 / size
        else:
            signal_power = self.signal_power

        noise_power = signal_power / self.snr

        if input_signal.dtype == complex:
            noise = np.sqrt(noise_power / 2) * (
                np.random.normal(size=size) + 1j * np.random.normal(size=size)
            )
        else:
            noise = np.sqrt(noise_power) * np.random.normal(size=size)

        return input_signal + noise
