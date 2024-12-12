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
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        input = np.asarray(input)
        if input.shape[-1] != self.code.length:
            raise ValueError("last dimension of 'input' should be the code length")
        return self._decode(input)

    @abstractmethod
    def _decode(self, input: npt.NDArray[Any]) -> npt.NDArray[np.integer]:
        pass
