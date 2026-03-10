from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._util import global_rng


@dataclass
class GaussianChannel:
    r"""
    Gaussian channel. It is defined by
    $$
        Y_n = X_n + Z_n,
    $$
    where $X_n$ is the channel *input signal*, $Y_n$ is the channel *output signal*, and $Z_n$ is the *noise*, which is iid according to a Gaussian distribution with zero mean. The channel is characterized by the *noise power* (variance)
    $$
        \sigma_Z^2 = \mathrm{E}[Z_n^2].
    $$

    The channel supports real- and complex-valued signals. In the complex case, the noise is circularly symmetric complex Gaussian, with the noise power
    equally divided between the real and imaginary parts.

    For more details, see <cite>CT06, Ch. 9</cite>.

    Parameters:
        noise_power: The noise power (variance) $\sigma_Z^2$. The default value is `0.0`, which corresponds to a noiseless channel.
    """

    noise_power: float = 0.0
    rng: np.random.Generator = field(default_factory=global_rng.get, repr=False)

    def transmit(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        r"""
        Transmits the input signal through the channel and returns the output signal.

        The input signal may be real- or complex-valued. If the input is real, the noise is real-valued Gaussian with variance equal to the noise power $\sigma_Z^2$. If the input is complex, the noise is circularly symmetric complex Gaussian, with the noise power equally divided between the real and imaginary parts, i.e., $\mathrm{E}[\mathrm{Re}\\{Z_n\\}^2] = \mathrm{E}[\mathrm{Im}\\{Z_n\\}^2] = \sigma_Z^2/2$.

        Parameters:
            input: The input signal $X_n$.

        Returns:
            output: The output signal $Y_n$.

        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> ch = komm.GaussianChannel(noise_power=0.025, rng=rng)
            >>> ch.transmit([1, 3, -3, -1, -1, 1, 3, 1, -1, 3]).round(2)
            array([ 1.05,  2.84, -2.88, -0.85, -1.31,  0.79,  3.02,  0.95, -1.  ,  2.87])

            >>> rng = np.random.default_rng(seed=42)
            >>> ch = komm.GaussianChannel(noise_power=0.05, rng=rng)
            >>> ch.transmit([1 + 3j, -3 - 1j, -1 + 1j, 3 + 1j, -1 + 3j]).round(2)
            array([ 1.05+2.84j, -2.88-0.85j, -1.31+0.79j,  3.02+0.95j, -1.  +2.87j])
        """
        input = np.array(input)

        if input.dtype == complex:
            noise_ri = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(self.noise_power / 2),
                size=(*input.shape, 2),
            )
            noise = noise_ri[..., 0] + 1j * noise_ri[..., 1]
        else:
            noise = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(self.noise_power),
                size=input.shape,
            )

        return input + noise
