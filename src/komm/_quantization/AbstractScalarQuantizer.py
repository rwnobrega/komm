from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class AbstractScalarQuantizer(ABC):
    @property
    @abstractmethod
    def levels(self) -> npt.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def num_levels(self) -> int:
        pass

    @abstractmethod
    def __call__(self, input_signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass
