from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import base
from .util import raised_cosine, rect


@dataclass
class RaisedCosinePulse(base.Pulse):
    r"""
    Raised-cosine pulse. For a given *roll-off factor* $\alpha$ satisfing $0 \leq \alpha \leq 1$, it is a [pulse](/ref/Pulse)
    with spectrum given by
    $$
        \hat{p}(f) = \begin{cases}
            1, & |f| \leq f_1, \\\\[1ex]
            \dfrac{1}{2} \left( 1 + \cos \left( \pi \dfrac{|f| - f_1}{f_2 - f_1}\right) \right), & f_1 \leq |f| \leq f_2, \\\\[1ex]
            0, & \text{otherwise}.
        \end{cases}
    $$
    where $f_1 = (1 - \alpha) / 2$ and $f_2 = (1 + \alpha) / 2$.

    The waveform of the raised-cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
      ![Raised-cosine pulse with roll-off factor 0.25.](/fig/pulse_raised_cosine_25.svg)
      </span>
      <span>
      ![Raised-cosine pulse with roll-off factor 0.75.](/fig/pulse_raised_cosine_75.svg)
      </span>
    </div>

    For more details, see [Wikipedia: Raised-cosine filter](https://en.wikipedia.org/wiki/Raised-cosine_filter).

    Notes:
        - For $\alpha = 0$ it reduces to the [sinc pulse](/ref/SincPulse).
        - For $\alpha = 1$ it becomes what is known as the _full cosine roll-off pulse_.

    Attributes:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `1.0`.
    """

    rolloff: float = 1.0

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the raised-cosine pulse, it is given by
        $$
            p(t) = \sinc(t) \frac{\cos(\pi \alpha t)}{1 - (2 \alpha t)^2}.
        $$

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.29 , 0.627, 0.897, 1.   , 0.897, 0.627, 0.29 , 0.   ])
        """
        return raised_cosine(t, self.rolloff)

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ))
            array([0. , 0. , 0.5, 1. , 1. , 1. , 0.5, 0. , 0. ])
        """
        α = self.rolloff
        f = np.asarray(f)
        if α == 0:
            return rect(f).astype(complex)
        f1 = (1 - α) / 2
        f2 = (1 + α) / 2
        band1 = abs(f) < f1
        band2 = (f1 <= abs(f)) * (abs(f) < f2)
        spectrum = np.zeros_like(f, dtype=complex)
        spectrum[band1] = 1.0
        spectrum[band2] = 0.5 * (1 + np.cos(np.pi * (abs(f[band2]) - f1) / α))
        return spectrum

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the raised-cosine pulse, it is given by
        $$
            R(\tau) = \sinc(\tau) \frac{\cos(\pi \alpha \tau)}{1 - (2 \alpha \tau)^2} - \frac{\alpha}{4} \sinc(\alpha \tau) \frac{\cos(\pi \tau)}{1 - (\alpha \tau)^2}.
        $$

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.06 , 0.334, 0.627, 0.853, 0.938, 0.853, 0.627, 0.334, 0.06 ])
        """
        α = self.rolloff
        tau = np.asarray(tau)
        if α == 0:
            return np.sinc(tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.sinc(α * tau) * np.cos(np.pi * tau) / (1 - (α * tau) ** 2)
        singularity = np.isclose(np.abs(tau), 1 / α)
        term[singularity] = np.cos(np.pi / α) / 2
        return raised_cosine(tau, α) - α / 4 * term

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the raised-cosine pulse, it is given by
        $$
            S(f) = \begin{cases}
                1, & |f| \leq f_1, \\\\[1ex]
                \dfrac{1}{4} \left( 1 + \cos \left( \pi \dfrac{|f| - f_1}{f_2 - f_1}\right) \right)^2, & f_1 \leq |f| \leq f_2, \\\\[1ex]
                0, & \text{otherwise}.
            \end{cases}
        $$

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.energy_density_spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0.  , 0.  , 0.25, 1.  , 1.  , 1.  , 0.25, 0.  , 0.  ])
        """
        return np.abs(self.spectrum(f)) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
