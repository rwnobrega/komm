from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IntegerCode(ABC):
    @abstractmethod
    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encode the input integer array.

        Parameters:
            input: The input integer array. It must be a one-dimensional array.

        Returns:
            output: The sequence of bits corresponding to the input integer array.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decode the input bit array.

        Parameters:
            input: The input bit array. It must be a one-dimensional array.

        Returns:
            output: The sequence of integers corresponding to the input bit array.
        """
        raise NotImplementedError
