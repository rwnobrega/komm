from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class SincPulse(base.Pulse):
    r"""
    Sinc pulse. It is a [pulse](/ref/Pulse) with spectrum given by
    $$
        \hat{h}(f) = \begin{cases}
            1, & -\frac{1}{2} \leq f < \frac{1}{2}, \\\\
            0, & \text{otherwise}.
        \end{cases}
    $$

    The waveform of the sinc pulse is depicted below.

    <figure markdown>
    ![Sinc pulse.](/figures/pulse_sinc.svg)
    </figure>

    For more details, see <cite>PS08, Sec. 9.2-1</cite>.

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the sinc pulse, it is given by
        $$
            h(t) = \sinc(t) = \frac{\sin(\pi t)}{\pi t}.
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> pulse.waveform([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]).round(4)
            array([0.3001, 0.6366, 0.9003, 1.    , 0.9003, 0.6366, 0.3001])
        """
        t = np.asarray(t)
        return np.sinc(t)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        For the sinc pulse, it is given by
        $$
            \hat{h}(f) = \rect(f).
        $$

        Examples:
            >>> pulse = komm.SincPulse()
            >>> np.abs(pulse.spectrum([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]))
            array([0., 1., 1., 1., 1., 0., 0.])
        """
        f = np.asarray(f)
        spectrum = 1.0 * (-0.5 <= f) * (f < 0.5)
        return spectrum.astype(complex)

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
