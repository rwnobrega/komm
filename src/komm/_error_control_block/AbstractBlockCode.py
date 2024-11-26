from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .._types import ArrayIntLike


class AbstractBlockCode(ABC):
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
    def enc_mapping(self, u: ArrayIntLike) -> npt.NDArray[np.int_]:
        pass

    @abstractmethod
    def inv_enc_mapping(self, v: ArrayIntLike) -> npt.NDArray[np.int_]:
        pass

    @abstractmethod
    def chk_mapping(self, r: ArrayIntLike) -> npt.NDArray[np.int_]:
        pass
