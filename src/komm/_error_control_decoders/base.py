from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .._error_control_block.base import BlockCode

T = TypeVar("T", bound=BlockCode)


class BlockDecoder(ABC, Generic[T]):
    code: T

    @abstractmethod
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Decode received words. This method takes one or more sequences of received words and returns their corresponding estimated message sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension contracted from $bn$ to $bk$, where $b$ is a positive integer.
        """
        raise NotImplementedError
