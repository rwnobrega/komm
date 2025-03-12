from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class GaussianPulse(base.Pulse):
    r"""
    Gaussian pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) = \mathrm{e}^{-\frac{1}{2} (2 \pi \bar{B} t)^2}
    $$
    where the $\bar{B} = B / \sqrt{\ln 2}$, and $B$ is the _half-power bandwidth_ of the filter. Its spectrum is given by
    $$
        \hat{h}(f) = \frac{1}{\sqrt{2 \pi} \bar{B}} \mathrm{e}^{-\frac{1}{2} (f / \bar{B})^2}.
    $$

    The Gaussian pulse is depicted below for $B = 0.5$, and for $B = 1$.

    <div class="centered" markdown>
      <span>
        ![Gaussian pulse with half-power bandwidth of 0.5.](/figures/pulse_gaussian_50.svg)
      </span>
      <span>
        ![Gaussian pulse with half-power bandwidth of 1.](/figures/pulse_gaussian_100.svg)
      </span>
    </div>

    Attributes:
        half_power_bandwidth: The half-power bandwidth $B$ of the pulse. The default value is `1.0`.

    Examples:
        >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.25)
        >>> pulse.waveform([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.3675, 0.6408, 0.8947, 1.    , 0.8947, 0.6408, 0.3675])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.0587, 0.3321, 0.9394, 1.3286, 0.9394, 0.3321, 0.0587])
    """

    half_power_bandwidth: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        t = np.asarray(t)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        return np.exp(-0.5 * (2 * np.pi * b_bar * t) ** 2)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        f = np.asarray(f)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        return 1 / (np.sqrt(2 * np.pi) * b_bar) * np.exp(-0.5 * (f / b_bar) ** 2)

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
