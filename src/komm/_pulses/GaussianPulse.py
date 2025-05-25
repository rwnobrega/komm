from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class GaussianPulse(base.Pulse):
    r"""
    Gaussian pulse. It is a [pulse](/ref/Pulse) with spectrum given by
    $$
        \hat{p}(f) = \frac{1}{\sqrt{2 \pi} \bar{B}} \mathrm{e}^{-\frac{1}{2} (f / \bar{B})^2},
    $$
    where the $\bar{B} = B / \sqrt{\ln 2}$, and $B$ is the _half-power bandwidth_ of the filter.

    The waveform of the Gaussian pulse is depicted below for $B = 0.5$, and for $B = 1$.

    <div class="centered" markdown>
      <span>
      ![Gaussian pulse with half-power bandwidth of 0.5.](/fig/pulse_gaussian_50.svg)
      </span>
      <span>
      ![Gaussian pulse with half-power bandwidth of 1.](/fig/pulse_gaussian_100.svg)
      </span>
    </div>

    Attributes:
        half_power_bandwidth: The half-power bandwidth $B$ of the pulse. The default value is `1.0`.
    """

    half_power_bandwidth: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Gaussian pulse, it is given by
        $$
            p(t) = \mathrm{e}^{-\frac{1}{2} (2 \pi \bar{B} t)^2}.
        $$

        Examples:
            >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.25)
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.169, 0.367, 0.641, 0.895, 1.   , 0.895, 0.641, 0.367, 0.169])
        """
        t = np.asarray(t)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        return np.exp(-0.5 * (2 * np.pi * b_bar * t) ** 2)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.25)
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.005, 0.059, 0.332, 0.939, 1.329, 0.939, 0.332, 0.059, 0.005])
        """
        f = np.asarray(f)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        spectrum = 1 / (np.sqrt(2 * np.pi) * b_bar) * np.exp(-0.5 * (f / b_bar) ** 2)
        return spectrum.astype(complex)

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Gaussian pulse, it is given by
        $$
            R(\tau) = \frac{1}{2 \sqrt{\pi} \bar{B}} \mathbb{e}^{-(\pi \bar{B} \tau)^2}.
        $$

        Examples:
            >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.25)
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.386, 0.569, 0.752, 0.889, 0.939, 0.889, 0.752, 0.569, 0.386])
        """
        tau = np.asarray(tau)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        return (1 / (2 * np.sqrt(np.pi) * b_bar)) * np.exp(
            -((np.pi * b_bar * tau) ** 2)
        )

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Gaussian pulse, it is given by
        $$
            S(f) = \frac{1}{2 \pi \bar{B}^2} \mathrm{e}^{-(f / \bar{B})^2}.
        $$

        Examples:
            >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.25)
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.003, 0.11 , 0.883, 1.765, 0.883, 0.11 , 0.003, 0.   ])
        """
        f = np.asarray(f)
        b_bar = self.half_power_bandwidth / np.sqrt(np.log(2))
        return (1 / (2 * np.pi * b_bar**2)) * np.exp(-((f / b_bar) ** 2))

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
