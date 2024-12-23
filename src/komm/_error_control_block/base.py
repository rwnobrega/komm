from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BlockCode(ABC):
    @property
    @abstractmethod
    def length(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def redundancy(self) -> int:
        pass

    @abstractmethod
    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass

    @abstractmethod
    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass

    @abstractmethod
    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass
