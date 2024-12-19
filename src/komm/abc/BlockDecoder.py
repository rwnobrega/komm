from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, final

import numpy as np
import numpy.typing as npt

from .. import abc

T = TypeVar("T", bound=abc.BlockCode)


@dataclass
class BlockDecoder(ABC, Generic[T]):
    code: T

    @final
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Decode received words. This method takes one or more sequences of received words and returns their corresponding estimated message sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension contracted from $bn$ to $bk$, where $b$ is a positive integer.
        """
        input = np.asarray(input)
        if input.shape[-1] % self.code.length != 0:
            raise ValueError(
                "last dimension of 'input' must be a multiple of code length"
                f" {self.code.length} (got {input.shape[-1]})"
            )
        r = input.reshape(*input.shape[:-1], -1, self.code.length)
        u_hat = self._decode(r)
        output = u_hat.reshape(*u_hat.shape[:-2], -1)
        return output

    @abstractmethod
    def _decode(self, r: npt.NDArray[Any]) -> npt.NDArray[np.integer]:
        pass
