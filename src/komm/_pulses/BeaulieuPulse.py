from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc
from .base import RootPulse
from .util import rect


@dataclass
class BeaulieuPulse(abc.Pulse):
    r"""
    Beaulieu pulse. For a given *roll-off factor* $\alpha$ satisfying $0 \leq \alpha \leq 1$, it is a [pulse](/ref/Pulse) with spectrum given by
    $$
        \hat{p}(f) = \begin{cases}
            1, & |f| \leq f_1, \\\\[1ex]
            \mathrm{e}^{-\beta (|f| - f_1)}, & f_1 \leq |f| \leq 1/2, \\\\[1ex]
            1 - \mathrm{e}^{-\beta (f_2 - |f|)}, & 1/2 \leq |f| \leq f_2, \\\\[1ex]
            0, & \text{otherwise},
        \end{cases}
    $$
    where $f_1 = (1 - \alpha) / 2$, $f_2 = (1 + \alpha) / 2$, and $\beta = (2 \ln 2) / \alpha$. It is also known as the *flipped-exponential (FE) pulse*, and as the *"better than raised-cosine" (BTRC) pulse*, since it is less sensitive to symbol timing errors than the [raised-cosine pulse](/ref/RaisedCosinePulse) with the same roll-off factor.

    The waveform of the Beaulieu pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
      ![Beaulieu pulse with roll-off factor 0.25.](/fig/pulse_beaulieu_25.svg)
      </span>
      <span>
      ![Beaulieu pulse with roll-off factor 0.75.](/fig/pulse_beaulieu_75.svg)
      </span>
    </div>

    For more details, see <cite>BTD01</cite>.

    Notes:
        - For $\alpha = 0$ it reduces to the [sinc pulse](/ref/SincPulse).

    Parameters:
        rolloff: The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$. The default value is `1.0`.
    """

    rolloff: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.rolloff <= 1:
            raise ValueError("'rolloff' must satisfy 0 <= rolloff <= 1")

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Beaulieu pulse, it is given by
        $$
            p(t) = \sinc(t) \frac{4 \beta \pi t \sin(\pi \alpha t) + 2 \beta^2 \cos(\pi \alpha t) - \beta^2}{(2 \pi t)^2 + \beta^2}.
        $$

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.   , 0.28 , 0.618, 0.894, 1.   , 0.894, 0.618, 0.28 , 0.   ])
        """
        α = self.rolloff
        t = np.asarray(t, dtype=float)
        if α == 0:
            return np.sinc(t)
        β = 2 * np.log(2) / α
        num = (
            4 * β * np.pi * t * np.sin(np.pi * α * t)
            + 2 * β**2 * np.cos(np.pi * α * t)
            - β**2
        )
        den = (2 * np.pi * t) ** 2 + β**2
        return np.sinc(t) * num / den

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ))
            array([0. , 0. , 0.5, 1. , 1. , 1. , 0.5, 0. , 0. ])
        """
        α = self.rolloff
        f = np.asarray(f)
        if α == 0:
            return rect(f).astype(complex)
        β = 2 * np.log(2) / α
        f1 = (1 - α) / 2
        f2 = (1 + α) / 2
        band1 = abs(f) < f1
        band2 = (f1 <= abs(f)) * (abs(f) < 0.5)
        band3 = (0.5 <= abs(f)) * (abs(f) < f2)
        spectrum = np.zeros_like(f, dtype=complex)
        spectrum[band1] = 1.0
        spectrum[band2] = np.exp(-β * (abs(f[band2]) - f1))
        spectrum[band3] = 1 - np.exp(-β * (f2 - abs(f[band3])))
        return spectrum

    def energy(self) -> float:
        r"""
        For the Beaulieu pulse, it is given by
        $$
            E = 1 - \frac{\alpha}{4 \ln 2}.
        $$

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> pulse.energy()  # doctest: +FLOAT_CMP
            0.9098315599444398
        """
        α = self.rolloff
        return float(1 - α / (4 * np.log(2)))

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Beaulieu pulse, it is given by
        $$
        \begin{aligned}
            R(\tau) = \sinc(\tau) \( 2 \cos(\pi \alpha \tau) - 1 \)
            & + \frac{4 \beta \( \cos(2 \pi f_1 \tau) + \cos(2 \pi f_2 \tau) \) + 4 \pi \tau \( \sin(2 \pi f_2 \tau) - \sin(2 \pi f_1 \tau) \) - 2 \beta \cos(\pi \tau)}{4 \beta^2 + (2 \pi \tau)^2} \\\\[1ex]
            & - \frac{4 \beta \cos(2 \pi f_2 \tau) + 8 \pi \tau \sin(2 \pi f_2 \tau) - 2 \beta \cos(\pi \tau) - 4 \pi \tau \sin(\pi \tau)}{\beta^2 + (2 \pi \tau)^2}.
        \end{aligned}
        $$

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> pulse.autocorrelation(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([0.084, 0.341, 0.618, 0.83 , 0.91 , 0.83 , 0.618, 0.341, 0.084])
        """
        α = self.rolloff
        τ = np.asarray(tau, dtype=float)
        if α == 0:
            return np.sinc(τ)
        β = 2 * np.log(2) / α
        f1 = (1 - α) / 2
        f2 = (1 + α) / 2
        c0, s0 = np.cos(np.pi * τ), np.sin(np.pi * τ)
        c1, s1 = np.cos(2 * np.pi * f1 * τ), np.sin(2 * np.pi * f1 * τ)
        c2, s2 = np.cos(2 * np.pi * f2 * τ), np.sin(2 * np.pi * f2 * τ)
        term0 = np.sinc(τ) * (2 * np.cos(np.pi * α * τ) - 1)
        term1 = (4 * β * (c1 + c2) + 4 * np.pi * τ * (s2 - s1) - 2 * β * c0) / (
            4 * β**2 + (2 * np.pi * τ) ** 2
        )
        term2 = (4 * β * c2 + 8 * np.pi * τ * s2 - 2 * β * c0 - 4 * np.pi * τ * s0) / (
            β**2 + (2 * np.pi * τ) ** 2
        )
        return term0 + term1 - term2

    def energy_spectral_density(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        For the Beaulieu pulse, it is given by
        $$
            S(f) = \begin{cases}
                1, & |f| \leq f_1, \\\\[1ex]
                \mathrm{e}^{-2 \beta (|f| - f_1)}, & f_1 \leq |f| \leq 1/2, \\\\[1ex]
                \( 1 - \mathrm{e}^{-\beta (f_2 - |f|)} \)^2, & 1/2 \leq |f| \leq f_2, \\\\[1ex]
                0, & \text{otherwise}.
            \end{cases}
        $$

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> pulse.energy_spectral_density(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )
            array([0.  , 0.  , 0.25, 1.  , 1.  , 1.  , 0.25, 0.  , 0.  ])
        """
        return np.abs(self.spectrum(f)) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        r"""
        For the Beaulieu pulse, the support is given by $(-\infty, \infty)$.

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
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
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25)
            >>> pulse.taps(samples_per_symbol=4, span=(-1, 1)).round(3)
            array([0.   , 0.28 , 0.618, 0.894, 1.   , 0.894, 0.618, 0.28 , 0.   ])
            >>> pulse.taps(samples_per_symbol=4, span=(-16, 16)).shape
            (129,)
        """
        return super().taps(samples_per_symbol, span)

    def root(self) -> abc.Pulse:
        r"""
        For the Beaulieu pulse, the waveform of the square-root version does not have a closed-form expression; it is computed here by numerical integration of the square root of the spectrum.

        Examples:
            >>> pulse = komm.BeaulieuPulse(rolloff=0.25).root()
            >>> pulse.waveform(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... ).round(3)
            array([-0.08 ,  0.215,  0.611,  0.953,  1.087,  0.953,  0.611,  0.215,
                   -0.08 ])
            >>> np.abs(pulse.spectrum(
            ...     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            ... )).round(3)
            array([0.   , 0.   , 0.707, 1.   , 1.   , 1.   , 0.707, 0.   , 0.   ])
        """
        return _RootBeaulieuPulse(self)


class _RootBeaulieuPulse(RootPulse[BeaulieuPulse]):
    @cached_property
    def _quadrature(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Gauss–Legendre nodes and weights on [0, sqrt(α/2)], used to integrate
        # the outer band of the square-root spectrum after the substitution
        # f = f2 - v², which removes the square-root cusp at f = f2.
        α = self.base_pulse.rolloff
        nodes, weights = np.polynomial.legendre.leggauss(256)
        v_max = np.sqrt(α / 2)
        return (nodes + 1) * (v_max / 2), weights * (v_max / 2)

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        # The waveform is the inverse Fourier transform of the square root of
        # the spectrum of the base pulse. The bands [0, f1] and [f1, 1/2] are
        # integrated in closed form; the band [1/2, f2] is integrated numerically.
        α = self.base_pulse.rolloff
        t = np.asarray(t, dtype=float)
        if α == 0:
            return np.sinc(t)
        b = np.log(2) / α  # half of β
        f1 = (1 - α) / 2
        f2 = (1 + α) / 2
        # Band [0, f1], where the square-root spectrum equals 1:
        band1 = (1 - α) * np.sinc((1 - α) * t)
        # Band [f1, 1/2], where the square-root spectrum equals e^{-b (f - f1)}:
        num = (
            b * np.cos(2 * np.pi * f1 * t)
            - 2 * np.pi * t * np.sin(2 * np.pi * f1 * t)
            - b / np.sqrt(2) * np.cos(np.pi * t)
            + np.sqrt(2) * np.pi * t * np.sin(np.pi * t)
        )
        band2 = 2 * num / (b**2 + (2 * np.pi * t) ** 2)
        # Band [1/2, f2], where the square-root spectrum equals
        # sqrt(1 - e^{-2b (f2 - f)}):
        v, w = self._quadrature
        g = 4 * w * v * np.sqrt(1 - np.exp(-2 * b * v**2))
        band3 = np.cos(2 * np.pi * t[..., np.newaxis] * (f2 - v**2)) @ g
        return band1 + band2 + band3
