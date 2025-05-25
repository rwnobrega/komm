from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .RaisedCosinePulse import RaisedCosinePulse
from .util import raised_cosine


@dataclass
class RootRaisedCosinePulse(base.Pulse):
    r"""
    Root-raised-cosine pulse. It is a [pulse](/ref/Pulse) whose spectrum is given by the square root of the spectrum of the [raised cosine pulse](/ref/RaisedCosinePulse) with same roll-off factor.

    The waveform of the root-raised cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
      ![Root-raised-cosine pulse with roll-off factor 0.25.](/fig/pulse_root_raised_cosine_25.svg)
      </span>
      <span>
      ![Root-raised-cosine pulse with roll-off factor 0.75.](/fig/pulse_root_raised_cosine_75.svg)
      </span>
    </div>

    For more details, see [Wikipedia: Root-raised-cosine filter](https://en.wikipedia.org/wiki/Root-raised-cosine_filter).

    Attributes:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `1.0`.
    """

    rolloff: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the root-raised-cosine pulse, it is given by
        $$
          p(t) = \frac{\sin \( 2 \pi f_1 t \) + 4 \alpha t \cos \( 2 \pi f_2 t \)}{\pi t \( 1 - (4 \alpha t)^2 \)},
        $$
        where $\alpha$ is the roll-off factor, $f_1 = (1 - \alpha) / 2$, and $f_2 = (1 + \alpha) / 2$.

        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([-0.064,  0.238,  0.622,  0.943,  1.068,  0.943,  0.622,  0.238,
                   -0.064])
        """
        α = self.rolloff
        t = np.asarray(t)
        if α == 0:
            return np.sinc(t)
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
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.   , 0.   , 0.707, 1.   , 1.   , 1.   , 0.707, 0.   , 0.   ])
        """
        return np.sqrt(RaisedCosinePulse(self.rolloff).spectrum(f))

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the root-raised-cosine pulse, it is given by
        $$
            R(\tau) = \sinc(\tau) \frac{\cos(\pi \alpha \tau)}{1 - (2 \alpha \tau)^2}.
        $$

        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.29 , 0.627, 0.897, 1.   , 0.897, 0.627, 0.29 , 0.   ])
        """
        return raised_cosine(tau, self.rolloff)

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the root-raised-cosine pulse, it is given by
        $$
            S(f) = \begin{cases}
                1, & |f| \leq f_1, \\\\[1ex]
                \dfrac{1}{2} \left( 1 + \cos \left( \pi \dfrac{|f| - f_1}{f_2 - f_1}\right) \right), & f_1 \leq |f| \leq f_2, \\\\[1ex]
                0, & \text{otherwise}.
            \end{cases}
        $$

        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25)
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0. , 0. , 0.5, 1. , 1. , 1. , 0.5, 0. , 0. ])
        """
        return np.abs(self.spectrum(f)) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
