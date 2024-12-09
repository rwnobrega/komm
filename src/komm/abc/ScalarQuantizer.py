from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ScalarQuantizer(ABC):
    @property
    @abstractmethod
    def levels(self) -> npt.NDArray[np.floating]:
        pass

    @property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.floating]:
        pass

    @property
    @abstractmethod
    def num_levels(self) -> int:
        pass

    @abstractmethod
    def __call__(self, input_signal: npt.ArrayLike) -> npt.NDArray[np.floating]:
        pass
