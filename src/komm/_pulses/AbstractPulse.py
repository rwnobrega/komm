from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class AbstractPulse(ABC):
    @abstractmethod
    def waveform(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def spectrum(self, f: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        pass
