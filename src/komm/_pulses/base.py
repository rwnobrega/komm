from abc import ABC, abstractmethod
from functools import cached_property

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
    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        The autocorrelation function $R(\tau)$ of the pulse.
        """
        raise NotImplementedError

    @abstractmethod
    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        The energy density spectrum $S(f)$ of the pulse.
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
