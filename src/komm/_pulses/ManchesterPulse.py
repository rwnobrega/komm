from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .util import rect, tri


@dataclass
class ManchesterPulse(base.Pulse):
    r"""
    Manchester pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        p(t) =
        \begin{cases}
            -1, & 0 \leq t <  1/2, \\\\
            1, & 1/2 \leq t < 1, \\\\
            0, & \text{otherwise},
        \end{cases}
    $$

    The waveform of the Manchester pulse is depicted below.

    <figure markdown>
    ![Manchester pulse.](/fig/pulse_manchester.svg)
    </figure>

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Manchester pulse, it is given by
        $$
            p(t) = -\rect\left(\frac{t - 1/4}{1/2}\right) + \rect\left(\frac{t - 3/4}{1/2}\right).
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([ 0.,  0.,  0.,  0., -1., -1.,  1.,  1.,  0.])
        """
        t = np.asarray(t)
        return -rect(2 * (t - 0.25)) + rect(2 * (t - 0.75))

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        For the Manchester pulse, it is given by
        $$
            \hat{p}(f) = \sinc \left( \frac{f}{2} \right) \, \sin \left( 2 \pi \frac{f}{4} \right) \mathrm{e}^{-\mathrm{j} 2 \pi (f/2\,+\,1/4)}.
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.637, 0.725, 0.637, 0.373, 0.   , 0.373, 0.637, 0.725, 0.637])
        """
        f = np.asarray(f)
        cexp = np.exp(-2j * np.pi * (f / 2 + 1 / 4))
        centered = np.sinc(f / 2) * np.sin(2 * np.pi * f / 4)
        return centered.astype(complex) * cexp

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Manchester pulse, it is given
        $$
            R(\tau) = \tri \left( \frac{\tau}{1/2} \right) - \frac{1}{2} \tri \left( \frac{\tau + 1/2}{1/2} \right) - \frac{1}{2} \tri \left( \frac{\tau - 1/2}{1/2} \right).
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([ 0.  , -0.25, -0.5 ,  0.25,  1.  ,  0.25, -0.5 , -0.25,  0.  ])
        """
        tau = np.asarray(tau)
        return tri(2 * tau) - 0.5 * tri(2 * (tau - 0.5)) - 0.5 * tri(2 * (tau + 0.5))

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Manchester pulse, it is given by
        $$
            S(f) = \sinc^2 \left( \frac{f}{2} \right) \, \sin^2 \left( 2 \pi \frac{f}{4} \right).
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.405, 0.525, 0.405, 0.139, 0.   , 0.139, 0.405, 0.525, 0.405])
        """
        f = np.asarray(f)
        return np.sinc(f / 2) ** 2 * np.sin(2 * np.pi * f / 4) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)
