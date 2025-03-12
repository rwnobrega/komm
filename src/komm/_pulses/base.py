from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt


class Pulse(ABC):
    @abstractmethod
    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    @abstractmethod
    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        raise NotImplementedError
