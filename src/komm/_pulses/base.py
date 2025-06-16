from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, final

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

    @final
    def support_kind(self) -> Literal["finite", "infinite", "semi-infinite"]:
        if self.support == (-np.inf, np.inf):
            return "infinite"
        elif self.support[0] != -np.inf and self.support[1] != np.inf:
            return "finite"
        else:
            return "semi-infinite"

    def time_span(self, truncation: int | None = None) -> tuple[int, int]:
        if self.support_kind() == "finite":
            if truncation is not None:
                raise ValueError("'truncation' only applies to infinite-support pulses")
            return int(np.floor(self.support[0])), int(np.ceil(self.support[1]))
        elif self.support_kind() == "infinite":
            truncation = truncation or 32
            if truncation <= 0 or truncation % 2 != 0:
                raise ValueError("'truncation' must be an even positive integer")
            return -truncation // 2, truncation // 2
        else:  # if self.support_kind() == "semi-infinite":
            raise ValueError("pulses with semi-infinite support are not supported")

    @final
    def taps(
        self,
        samples_per_symbol: int,
        truncation: int | None = None,
    ) -> npt.NDArray[np.floating]:
        sps = samples_per_symbol
        start, end = self.time_span(truncation=truncation)
        t = np.arange(start * sps, end * sps + 1) / sps
        return self.waveform(t)
