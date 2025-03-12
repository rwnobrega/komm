from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class RectangularPulse(base.Pulse):
    r"""
    Rectangular pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) =
        \begin{cases}
            1, & 0 \leq t < w, \\\\
            0, & \text{otherwise}.
        \end{cases}
        = \mathrm{rect}\left(\frac{t}{w}\right),
    $$
    where $w$ is the *width* of the pulse, which must satisfy $0 \leq w \leq 1$. Its spectrum is given by
    $$
        \hat{h}(f) = w \, \operatorname{sinc}(w f).
    $$

    The rectangular pulse is depicted below for $w = 1$, and for $w = 0.5$.

    <div class="centered" markdown>
      <span>
        ![Rectangular NRZ pulse.](/figures/pulse_rectangular_nrz.svg)
      </span>
      <span>
        ![Rectangular RZ pulse.](/figures/pulse_rectangular_rz.svg)
      </span>
    </div>

    Notes:
        - For $w = 1$ it is also called the _NRZ pulse_.
        - For $w = 0.5$ it is also called the _halfway RZ pulse_.

    Attributes:
        width: The width $w$ of the pulse. Must satisfy $0 \leq w \leq 1$. The default value is `1.0`.

    Examples:
        >>> pulse = komm.RectangularPulse(width=1.0)  # NRZ pulse
        >>> pulse.waveform([-0.50, -0.25,  0.00,  0.25,  0.50,  0.75,  1.00])
        array([0., 0., 1., 1., 1., 1., 0.])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.3001, 0.6366, 0.9003, 1.    , 0.9003, 0.6366, 0.3001])

        >>> pulse = komm.RectangularPulse(width=0.5)  # Halfway RZ pulse
        >>> pulse.waveform([-0.50, -0.25,  0.00,  0.25,  0.50,  0.75,  1.00])
        array([0., 0., 1., 1., 0., 0., 0.])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.3921, 0.4502, 0.4872, 0.5   , 0.4872, 0.4502, 0.3921])
    """

    width: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        t = np.asarray(t)
        return 1.0 * (0.0 <= t) * (t < self.width)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        f = np.asarray(f)
        return self.width * np.sinc(self.width * f)

    @cached_property
    def support(self) -> tuple[float, float]:
        return (0.0, self.width)
