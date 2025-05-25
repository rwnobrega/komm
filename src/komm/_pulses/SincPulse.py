from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .util import rect


@dataclass
class SincPulse(base.Pulse):
    r"""
    Sinc pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        p(t) = \frac{\sin(\pi t)}{\pi t} = \sinc(t).
    $$

    The waveform of the sinc pulse is depicted below.

    <figure markdown>
    ![Sinc pulse.](/fig/pulse_sinc.svg)
    </figure>

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the sinc pulse, it is given by
        $$
            p(t) = \sinc(t).
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.3  , 0.637, 0.9  , 1.   , 0.9  , 0.637, 0.3  , 0.   ])
        """
        t = np.asarray(t)
        return np.sinc(t)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        For the sinc pulse, it is given by
        $$
            \hat{p}(f) = \rect(f).
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ))
            array([0., 0., 1., 1., 1., 1., 0., 0., 0.])
        """
        f = np.asarray(f)
        return rect(f).astype(complex)

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the sinc pulse, it is given by
        $$
            R(\tau) = \sinc(\tau).
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.3  , 0.637, 0.9  , 1.   , 0.9  , 0.637, 0.3  , 0.   ])
        """
        tau = np.asarray(tau)
        return np.sinc(tau)

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the sinc pulse, it is given by
        $$
            S(f) = \rect(f).
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0., 0., 1., 1., 1., 1., 0., 0., 0.])
        """
        f = np.asarray(f)
        return rect(f)

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
