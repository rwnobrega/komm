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
    def enc_mapping(self, u: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass

    @abstractmethod
    def inv_enc_mapping(self, v: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass

    @abstractmethod
    def chk_mapping(self, r: npt.ArrayLike) -> npt.NDArray[np.integer]:
        pass
