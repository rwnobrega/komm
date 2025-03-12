from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .RaisedCosinePulse import RaisedCosinePulse


@dataclass
class RootRaisedCosinePulse(base.Pulse):
    r"""
    Root raised cosine pulse. It is a [pulse](/ref/Pulse) with waveform given by
    $$
        h(t) = \frac{\sin \( 2 \pi f_1 t \) + 4 \alpha t \cos \( 2 \pi f_2 t \)}{\pi t \( 1 - (4 \alpha t)^2 \)},
    $$
    where $\alpha$ is the *roll-off factor* (which must satisfy $0 \leq \alpha \leq 1$), $f_1 = (1 - \alpha) / 2$, and $f_2 = (1 + \alpha) / 2$. Its spectrum is given by the square root of the spectrum of the [raised cosine pulse](/ref/RaisedCosinePulse).

    The root raised cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
        ![Root raised cosine pulse with roll-off factor 0.25.](/figures/pulse_root_raised_cosine_25.svg)
      </span>
      <span>
        ![Root raised cosine pulse with roll-off factor 0.75.](/figures/pulse_root_raised_cosine_75.svg)
      </span>
    </div>

    Attributes:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `0.0`.

    Examples:
        >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
        >>> pulse.waveform([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.2379, 0.6218, 0.9432, 1.0683, 0.9432, 0.6218, 0.2379])
        >>> pulse.spectrum([-0.75, -0.50, -0.25,  0.00,  0.25,  0.50,  0.75]).round(4)
        array([0.    , 0.7071, 1.    , 1.    , 1.    , 0.7071, 0.    ])
    """

    rolloff: float = 0.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        a = self.rolloff
        t = np.asarray(t) + 1e-8  # TODO: Improve this workaround
        f1 = (1 - a) / 2
        f2 = (1 + a) / 2
        num = np.sin(2 * np.pi * f1 * t) + 4 * a * t * np.cos(2 * np.pi * f2 * t)
        den = np.pi * t * (1 - (4 * a * t) ** 2)
        return num / den

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        a = self.rolloff
        hf_rc = RaisedCosinePulse(rolloff=a).spectrum
        return np.sqrt(hf_rc(f))

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
