from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .RaisedCosinePulse import RaisedCosinePulse
from .SincPulse import SincPulse


@dataclass
class RootRaisedCosinePulse(base.Pulse):
    r"""
    Root-raised-cosine pulse. It is a [pulse](/ref/Pulse) whose spectrum is given by the square root of the spectrum of the [raised cosine pulse](/ref/RaisedCosinePulse) with same roll-off factor.

    The waveform of the root-raised cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
      ![Root-raised-cosine pulse with roll-off factor 0.25.](/figures/pulse_root_raised_cosine_25.svg)
      </span>
      <span>
      ![Root-raised-cosine pulse with roll-off factor 0.75.](/figures/pulse_root_raised_cosine_75.svg)
      </span>
    </div>

    For more details, see [Wikipedia: Root-raised-cosine filter](https://en.wikipedia.org/wiki/Root-raised-cosine_filter).

    Attributes:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `0.0`.
    """

    rolloff: float = 0.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the root-raised-cosine pulse, it is given by
        $$
          h(t) = \frac{\sin \( 2 \pi f_1 t \) + 4 \alpha t \cos \( 2 \pi f_2 t \)}{\pi t \( 1 - (4 \alpha t)^2 \)},
        $$
        where $\alpha$ is the roll-off factor, $f_1 = (1 - \alpha) / 2$, and $f_2 = (1 + \alpha) / 2$.

        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
            >>> pulse.waveform([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]).round(4)
            array([0.2379, 0.6218, 0.9432, 1.0683, 0.9432, 0.6218, 0.2379])
        """
        α = self.rolloff
        t = np.asarray(t)
        if α == 0:
            return SincPulse().waveform(t)
        f1 = (1 - α) / 2
        f2 = (1 + α) / 2
        num = np.sin(2 * np.pi * f1 * t) + 4 * α * t * np.cos(2 * np.pi * f2 * t)
        den = np.pi * t * (1 - (4 * α * t) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            waveform = num / den
        singularity = np.isclose(t, 0)
        waveform[singularity] = 1 + α * (4 / np.pi - 1)
        singularity = np.isclose(np.abs(t), 1 / (4 * α))
        lhs = (1 + 2 / np.pi) * np.sin(np.pi / (4 * α))
        rhs = (1 - 2 / np.pi) * np.cos(np.pi / (4 * α))
        waveform[singularity] = α / np.sqrt(2) * (lhs + rhs)
        return waveform

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
            >>> np.abs(pulse.spectrum(
            ...     [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            ... )).round(4)
            array([0.    , 0.7071, 1.    , 1.    , 1.    , 0.7071, 0.    ])
        """
        α = self.rolloff
        return np.sqrt(RaisedCosinePulse(rolloff=α).spectrum(f))

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
