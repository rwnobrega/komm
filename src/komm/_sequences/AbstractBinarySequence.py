from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt


class AbstractBinarySequence(ABC):
    @property
    @abstractmethod
    def bit_sequence(self) -> npt.NDArray[np.int_]:
        pass

    @property
    @abstractmethod
    def polar_sequence(self) -> npt.NDArray[np.int_]:
        pass

    @property
    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def autocorrelation(
        self, shifts: Optional[Sequence[int]], normalized: bool
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def cyclic_autocorrelation(
        self, shifts: Optional[Sequence[int]], normalized: bool
    ) -> npt.NDArray[np.float64]:
        pass
