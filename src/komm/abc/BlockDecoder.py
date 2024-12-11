from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, final

import numpy as np
import numpy.typing as npt

from .. import abc

TCode = TypeVar("TCode", bound=abc.BlockCode)


@dataclass
class BlockDecoder(ABC, Generic[TCode]):
    code: TCode

    @final
    def __call__(self, r: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r = np.asarray(r)
        if r.shape[-1] != self.code.length:
            raise ValueError("last dimension of 'r' should be the code length")
        return self._decode(r)

    @abstractmethod
    def _decode(self, r: npt.NDArray[Any]) -> npt.NDArray[np.integer]:
        pass
