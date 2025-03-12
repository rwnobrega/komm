from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IntegerCode(ABC):
    @abstractmethod
    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        raise NotImplementedError
