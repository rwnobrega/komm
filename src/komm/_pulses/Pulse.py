from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc
from ..types import Array1D


class Pulse(abc.Pulse):
    r"""
    General pulse [Not implemented yet].
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        raise NotImplementedError

    def energy(self) -> float:
        raise NotImplementedError

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def energy_spectral_density(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    @cached_property
    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    def taps(
        self, samples_per_symbol: int, span: tuple[int, int] | None = None
    ) -> Array1D[np.floating]:
        return super().taps(samples_per_symbol, span)
