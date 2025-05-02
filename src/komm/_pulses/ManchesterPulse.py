from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class ManchesterPulse(base.Pulse):
    r"""
    Manchester pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) =
        \begin{cases}
            -1, & 0 \leq t <  1/2, \\\\
            1, & 1/2 \leq t < 1, \\\\
            0, & \text{otherwise},
        \end{cases}
    $$

    The waveform of the Manchester pulse is depicted below.

    <figure markdown>
    ![Manchester pulse.](/figures/pulse_manchester.svg)
    </figure>

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Manchester pulse, it is given by
        $$
            h(t) = -\rect\left(\frac{t - 1/4}{1/2}\right) + \rect\left(\frac{t - 3/4}{1/2}\right).
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> pulse.waveform([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
            array([ 0.,  0., -1., -1.,  1.,  1.,  0.])
        """
        t = np.asarray(t)
        return -1.0 * (0 <= t) * (t < 0.5) + 1.0 * (0.5 <= t) * (t < 1.0)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        For the Manchester pulse, it is given by
        $$
            \hat{h}(f) = \sinc \left( \frac{f}{2} \right) \, \sin \left( 2 \pi \frac{f}{4} \right)  \mathrm{e}^{-\mathrm{j} 2 \pi (f/2\,+\,1/4)}.
        $$

        Examples:
            >>> pulse = komm.ManchesterPulse()
            >>> np.abs(pulse.spectrum(
            ...     [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            ... )).round(4)
            array([0.7245, 0.6366, 0.3729, 0.    , 0.3729, 0.6366, 0.7245])
        """
        f = np.asarray(f)
        cexp = np.exp(-2j * np.pi * (f / 2 + 1 / 4))
        centered = np.sinc(f / 2) * np.sin(2 * np.pi * f / 4)
        return centered.astype(complex) * cexp

    @cached_property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)
