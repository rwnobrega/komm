from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass
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

    Attributes:
        signal_power: The input signal power $P$. If equal to the string `'measured'`, then every time the channel is invoked the input signal power will be computed from the input itself (i.e., its squared Euclidean norm).

        snr: The channel signal-to-noise ratio $\snr$ (linear, not decibel). The default value is `np.inf`, which corresponds to a noiseless channel.
    """

    signal_power: float | Literal["measured"]
    snr: float = np.inf
    rng: np.random.Generator = field(default=np.random.default_rng(), repr=False)

    @cached_property
    def noise_power(self) -> float:
        r"""
        The noise power $N$.
        """
        if self.signal_power == "measured":
            raise ValueError(
                "noise power cannot be calculated when 'signal_power' is 'measured'"
            )
        return self.signal_power / self.snr

    def capacity(self) -> float:
        r"""
        Returns the channel capacity $C$. It is given by $C = \frac{1}{2}\log_2(1 + \snr)$, in bits per dimension.

        Examples:
            >>> awgn = komm.AWGNChannel(signal_power=1.0, snr=63.0)
            >>> awgn.capacity()
            np.float64(3.0)
        """
        return 0.5 * np.log1p(self.snr) / np.log(2.0)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Transmits the input signal through the channel and returns the output signal.

        Parameters:
            input: The input signal $X_n$.

        Returns:
            output: The output signal $Y_n$.

        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> awgn = komm.AWGNChannel(signal_power=5.0, snr=200.0, rng=rng)
            >>> x = [1.0, 3.0, -3.0, -1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 3.0]
            >>> awgn(x).round(2)
            array([ 1.05,  2.84, -2.88, -0.85, -1.31,  0.79,  3.02,  0.95, -1.  ,  2.87])
        """
        input = np.array(input)

        if self.signal_power == "measured":
            signal_power = np.linalg.norm(input) ** 2 / input.shape[-1]
        else:
            signal_power = self.signal_power

        noise_power = signal_power / self.snr

        if input.dtype == complex:
            noise = np.sqrt(noise_power / 2) * (
                self.rng.standard_normal(size=input.shape)
                + 1j * self.rng.standard_normal(size=input.shape)
            )
        else:
            noise = np.sqrt(noise_power) * self.rng.standard_normal(size=input.shape)

        return input + noise
