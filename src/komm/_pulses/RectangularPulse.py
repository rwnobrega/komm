from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .util import rect, tri


@dataclass
class RectangularPulse(base.Pulse):
    r"""
    Rectangular pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        p(t) =
        \begin{cases}
            1, & 0 \leq t < w, \\\\
            0, & \text{otherwise},
        \end{cases}
    $$
    where $w$ is the *relative width* of the pulse, which must satisfy $0 < w \leq 1$.

    The waveform of the rectangular pulse is depicted below for $w = 1$, and for $w = 0.5$.

    <div class="centered" markdown>
      <span>
      ![Rectangular NRZ pulse.](/fig/pulse_rectangular_nrz.svg)
      </span>
      <span>
      ![Rectangular RZ pulse.](/fig/pulse_rectangular_rz.svg)
      </span>
    </div>

    Notes:
        - For $w = 1$ it is also called the _NRZ pulse_.
        - For $w = 0.5$ it is also called the _halfway RZ pulse_.

    Attributes:
        width: The relative width $w$ of the pulse. Must satisfy $0 < w \leq 1$. The default value is `1.0`.
    """

    width: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the rectangular pulse, it is given by
        $$
            p(t) = \rect\left(\frac{t - w/2}{w}\right).
        $$

        Examples:
            >>> pulse = komm.RectangularPulse(width=1.0)  # NRZ pulse
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0., 0., 0., 0., 1., 1., 1., 1., 0.])

            >>> pulse = komm.RectangularPulse(width=0.5)  # Halfway RZ pulse
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0., 0., 0., 0., 1., 1., 0., 0., 0.])
        """
        w = self.width
        t = np.asarray(t)
        return rect((t - w / 2) / w)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        For the rectangular pulse, it is given by
        $$
            \hat{p}(f) = w \sinc(w f) \mathrm{e}^{-\mathrm{j} 2 \pi (w/2) f}.
        $$

        Examples:
            >>> pulse = komm.RectangularPulse(width=1.0)  # NRZ pulse
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.   , 0.3  , 0.637, 0.9  , 1.   , 0.9  , 0.637, 0.3  , 0.   ])

            >>> pulse = komm.RectangularPulse(width=0.5)  # Halfway RZ pulse
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.318, 0.392, 0.45 , 0.487, 0.5  , 0.487, 0.45 , 0.392, 0.318])
        """
        w = self.width
        f = np.asarray(f)
        cexp = np.exp(-2j * np.pi * w / 2 * f)
        centered = w * np.sinc(w * f)
        return centered.astype(complex) * cexp

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the rectangular pulse, it is given by
        $$
            R(\tau) = w \tri\left(\frac{\tau}{w}\right).
        $$

        Examples:
            >>> pulse = komm.RectangularPulse(width=1.0)  # NRZ pulse
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0.  , 0.25, 0.5 , 0.75, 1.  , 0.75, 0.5 , 0.25, 0.  ])

            >>> pulse = komm.RectangularPulse(width=0.5)  # Halfway RZ pulse
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0.  , 0.  , 0.  , 0.25, 0.5 , 0.25, 0.  , 0.  , 0.  ])
        """
        w = self.width
        tau = np.asarray(tau)
        return w * tri(tau / w)

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the rectangular pulse, it is given by
        $$
            S(f) = w^2 \sinc^2(w f).
        $$

        Examples:
            >>> pulse = komm.RectangularPulse(width=1.0)  # NRZ pulse
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.09 , 0.405, 0.811, 1.   , 0.811, 0.405, 0.09 , 0.   ])

            >>> pulse = komm.RectangularPulse(width=0.5)  # Halfway RZ pulse
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.101, 0.154, 0.203, 0.237, 0.25 , 0.237, 0.203, 0.154, 0.101])
        """
        w = self.width
        f = np.asarray(f)
        return w**2 * np.sinc(w * f) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (0.0, self.width)
