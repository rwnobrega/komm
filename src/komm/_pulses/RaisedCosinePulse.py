from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc
from .base import RootPulse
from .util import raised_cosine, rect


@dataclass
class RaisedCosinePulse(abc.Pulse):
    r"""
    Raised-cosine pulse. For a given *roll-off factor* $\alpha$ satisfying  $0 \leq \alpha \leq 1$, it is a [pulse](/ref/Pulse)
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

    For more details, see [Wikipedia: Raised-cosine filter](https://en.wikipedia.org/wiki/Raised-cosine_filter) and [Wikipedia: Root-raised-cosine filter](https://en.wikipedia.org/wiki/Root-raised-cosine_filter).

    Notes:
        - For $\alpha = 0$ it reduces to the [sinc pulse](/ref/SincPulse).
        - For $\alpha = 1$ it becomes what is known as the _full cosine roll-off pulse_.

    Parameters:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `1.0`.
    """

    rolloff: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.rolloff <= 1:
            raise ValueError("'rolloff' must satisfy 0 <= rolloff <= 1")

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

    def energy(self) -> float:
        r"""
        For the raised-cosine pulse, it is given by
        $$
            E = 1 - \frac{\alpha}{4}.
        $$

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.energy()
            0.9375
        """
        α = self.rolloff
        return 1 - α / 4

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
        tau = np.asarray(tau, dtype=float)
        scalar_input = tau.ndim == 0
        tau = np.atleast_1d(tau)
        if α == 0:
            result = np.sinc(tau)
            return result[0] if scalar_input else result
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.sinc(α * tau) * np.cos(np.pi * tau) / (1 - (α * tau) ** 2)
        singularity = np.isclose(np.abs(tau), 1 / α)
        term[singularity] = np.cos(np.pi / α) / 2
        result = raised_cosine(tau, α) - α / 4 * term
        return result[0] if scalar_input else result

    def energy_spectral_density(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
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
            >>> pulse.energy_spectral_density(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0.  , 0.  , 0.25, 1.  , 1.  , 1.  , 0.25, 0.  , 0.  ])
        """
        return np.abs(self.spectrum(f)) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        r"""
        For the raised-cosine pulse, the support is given by $(-\infty, \infty)$.

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.support
            (-inf, inf)
        """
        return (-np.inf, np.inf)

    def taps(
        self,
        samples_per_symbol: int,
        span: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25)
            >>> pulse.taps(samples_per_symbol=4, span=(-1, 1)).round(3)
            array([0.   , 0.29 , 0.627, 0.897, 1.   , 0.897, 0.627, 0.29 , 0.   ])
            >>> pulse.taps(samples_per_symbol=4, span=(-16, 16)).shape
            (129,)
        """
        return super().taps(samples_per_symbol, span)

    def root(self) -> abc.Pulse:
        r"""
        For the raised-cosine pulse, the square-root version is known as the *root-raised-cosine (RRC)* pulse, whose waveform is given by
        $$
          p(t) = \frac{\sin \( 2 \pi f_1 t \) + 4 \alpha t \cos \( 2 \pi f_2 t \)}{\pi t \( 1 - (4 \alpha t)^2 \)}.
        $$

        Examples:
            >>> pulse = komm.RaisedCosinePulse(rolloff=0.25).root()
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([-0.064,  0.238,  0.622,  0.943,  1.068,  0.943,  0.622,  0.238,
                   -0.064])
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.   , 0.   , 0.707, 1.   , 1.   , 1.   , 0.707, 0.   , 0.   ])
        """
        return _RootRaisedCosinePulse(self)


class _RootRaisedCosinePulse(RootPulse[RaisedCosinePulse]):
    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        α = self.base_pulse.rolloff
        t = np.asarray(t, dtype=float)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        if α == 0:
            result = np.sinc(t)
            return result[0] if scalar_input else result
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
        return waveform[0] if scalar_input else waveform
