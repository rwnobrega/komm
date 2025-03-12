from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class SincPulse(base.Pulse):
    r"""
    Sinc pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) = \operatorname{sinc}(t) = \frac{\sin(\pi t)}{\pi t},
    $$
    and spectrum given by
    $$
        \hat{h}(f) = \begin{cases}
            1, & |f| < \frac{1}{2}, \\\\
            0, & \text{otherwise}.
        \end{cases}
    $$

    The sinc pulse is depicted below.

    <figure markdown>
      ![Sinc pulse.](/figures/pulse_sinc.svg)
    </figure>

    For more details, see <cite>PS08, Sec. 9.2-1</cite>.

    **Attributes:**

    <span style="font-size: 90%; font-style: italic; color: gray; margin-left: 1em;">(No attributes)</span>

    Examples:
        >>> pulse = komm.SincPulse()
        >>> pulse.waveform([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.3001, 0.6366, 0.9003, 1.    , 0.9003, 0.6366, 0.3001])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75])
        array([0., 0., 1., 1., 1., 0., 0.])
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        t = np.asarray(t)
        return np.sinc(t)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        f = np.asarray(f)
        return 1.0 * (abs(f) < 0.5)

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
