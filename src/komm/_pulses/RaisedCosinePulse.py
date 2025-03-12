from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base


@dataclass
class RaisedCosinePulse(base.Pulse):
    r"""
    Raised cosine pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) = \operatorname{sinc}(t) \frac{\cos(\pi \alpha t)}{1 - (2 \alpha t)^2},
    $$
    where $\alpha$ is the *roll-off factor*, which must satisfy $0 \leq \alpha \leq 1$. Its spectrum is given by
    $$
        \hat{h}(f) = \begin{cases}
            1, & |f| \leq f_1, \\\\[1ex]
            \dfrac{1}{2} \left( 1 + \cos \left( \pi \dfrac{|f| - f_1}{f_2 - f_1}\right) \right), & f_1 \leq |f| \leq f_2, \\\\[1ex]
            0, & \text{otherwise}.
        \end{cases}
    $$
    where $f_1 = (1 - \alpha) / 2$ and $f_2 = (1 + \alpha) / 2$.

    The raised cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
        ![Raised cosine pulse with roll-off factor 0.25.](/figures/pulse_raised_cosine_25.svg)
      </span>
      <span>
        ![Raised cosine pulse with roll-off factor 0.75.](/figures/pulse_raised_cosine_75.svg)
      </span>
    </div>

    For more details, see <cite>PS08, Sec. 9.2-1</cite>.

    Notes:
        - For $\alpha = 0$ it reduces to the [sinc pulse](/ref/SincPulse).
        - For $\alpha = 1$ it becomes what is known as the _full cosine roll-off pulse_.

    Attributes:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `1.0`.

    Examples:
        >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
        >>> pulse.waveform([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.2904, 0.6274, 0.897 , 1.    , 0.897 , 0.6274, 0.2904])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75])
        array([0. , 0.5, 1. , 1. , 1. , 0.5, 0. ])
    """

    rolloff: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        a = self.rolloff
        t = np.asarray(t) + 1e-8  # TODO: Improve this workaround
        return np.sinc(t) * np.cos(np.pi * a * t) / (1 - (2 * a * t) ** 2)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        a = self.rolloff
        f = np.asarray(f)
        if a == 0:
            return 1.0 * (abs(f) < 0.5)
        f1 = (1 - a) / 2
        f2 = (1 + a) / 2
        band1 = abs(f) < f1
        band2 = (f1 < abs(f)) * (abs(f) < f2)
        return 1.0 * band1 + (0.5 * (1 + np.cos(np.pi * (abs(f) - f1) / a))) * band2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
