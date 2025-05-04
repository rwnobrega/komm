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
        raise NotImplementedError
