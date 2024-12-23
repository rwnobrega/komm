from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Pulse(ABC):
    @abstractmethod
    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        pass

    @abstractmethod
    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.floating]:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        pass
