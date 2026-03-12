from functools import cached_property

import numpy as np
import numpy.typing as npt

from .. import abc


class Pulse(abc.Pulse):
    r"""
    General pulse [Not implemented yet].
    """

    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        raise NotImplementedError

    def autocorrelation(self, tau: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def energy_density_spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    @cached_property
    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    def taps(
        self, samples_per_symbol: int, span: tuple[int, int] | None = None
    ) -> npt.NDArray[np.floating]:
        return super().taps(samples_per_symbol, span)
