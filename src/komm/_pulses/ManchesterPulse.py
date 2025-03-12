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
    and spectrum given by
    $$
        \hat{h}(f) = \operatorname{sinc}^2 \left( \frac{f}{2} \right) \, \sin^2 \left( \frac{\pi f}{2} \right).
    $$

    The Manchester pulse is depicted below.

    <figure markdown>
      ![Manchester pulse.](/figures/pulse_manchester.svg)
    </figure>

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>

    Examples:
        >>> pulse = komm.ManchesterPulse()
        >>> pulse.waveform([-0.50, -0.25,  0.00,  0.25,  0.50,  0.75,  1.00])
        array([ 0.,  0., -1., -1.,  1.,  1.,  0.])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.5249, 0.4053, 0.1391, 0.    , 0.1391, 0.4053, 0.5249])
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        t = np.asarray(t)
        return -1.0 * (0 <= t) * (t < 0.5) + 1.0 * (0.5 <= t) * (t < 1.0)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        f = np.asarray(f)
        return np.sinc(f / 2) ** 2 * np.sin(np.pi * f / 2) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)
