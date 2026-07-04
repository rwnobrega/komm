from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt


class Pulse(ABC):
    @abstractmethod
    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        The waveform $p(t)$ of the pulse.
        """
        raise NotImplementedError

    @abstractmethod
    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        r"""
        The spectrum $\hat{p}(f)$ of the pulse.
        """
        raise NotImplementedError

    @abstractmethod
    def energy(self) -> float:
        r"""
        The energy $E$ of the pulse.
        """
        raise NotImplementedError

    @abstractmethod
    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        The autocorrelation function $R(\tau)$ of the pulse.
        """
        raise NotImplementedError

    @abstractmethod
    def energy_spectral_density(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        The energy spectral density $S(f)$ of the pulse.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        r"""
        The support of the pulse waveform $p(t)$, defined as the interval $[a, b]$ where $p(t)$ is non-zero.
        """
        raise NotImplementedError

    @abstractmethod
    def taps(
        self,
        samples_per_symbol: int,
        span: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Returns the FIR taps of the pulse.

        Parameters:
            samples_per_symbol: The number of samples per symbol.
            span: The time span to consider for the taps. This parameter is optional for pulses with finite support (defaults to $[0, 1]$), but required for pulses with infinite support.
        """
        if span is None and (self.support[0] == -np.inf or self.support[1] == np.inf):
            raise ValueError(
                "pulses with infinite support require 'span' to be specified"
            )
        sps = samples_per_symbol
        start, end = span or (0, 1)
        t = np.arange(start * sps, end * sps + 1) / sps
        return self.waveform(t)

    def root(self) -> "Pulse":
        r"""
        Returns the square-root version of the pulse, defined as the pulse whose spectrum is given by the square root of the spectrum of the original pulse. This method is only implemented for Nyquist pulses.
        """
        raise NotImplementedError("'root' is only implemented for Nyquist pulses")


T = TypeVar("T", bound=Pulse)


@dataclass
class RootPulse(Pulse, Generic[T]):
    r"""
    Abstract base class for square-root pulses. A *square-root pulse* is a pulse whose spectrum is given by the square root of the spectrum of a given base pulse; the latter must be band-limited with real and non-negative spectrum (e.g., a Nyquist pulse). Concrete subclasses must implement the `waveform` method.
    """

    base_pulse: T

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        return np.sqrt(self.base_pulse.spectrum(f))

    def energy(self) -> float:
        # By Parseval, E = ∫ |p̂(f)|² df = ∫ q̂(f) df = q(0), where q is the base pulse.
        return float(self.base_pulse.waveform(0.0))

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        # The ESD of the square-root pulse equals the spectrum of the base pulse, so
        # the autocorrelation of the former equals the waveform of the latter.
        return self.base_pulse.waveform(tau)

    def energy_spectral_density(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return np.abs(self.spectrum(f)) ** 2

    @cached_property
    def support(self) -> tuple[float, float]:
        # The square-root pulse is band-limited, so its waveform has infinite support.
        return (-np.inf, np.inf)

    def taps(
        self,
        samples_per_symbol: int,
        span: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.floating]:
        return super().taps(samples_per_symbol, span)

    def __repr__(self) -> str:
        return f"{self.base_pulse!r}.root()"
