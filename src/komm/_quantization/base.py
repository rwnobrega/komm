from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt


class ScalarQuantizer(ABC):
    @cached_property
    @abstractmethod
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer levels $v_0, v_1, \ldots, v_{L-1}$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$.
        """
        raise NotImplementedError

    @abstractmethod
    def digitize(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Returns the quantization indices for the input signal.

        Parameters:
            input: The input signal $x$ to be digitized.

        Returns:
            output: The integer indices of the quantization levels for each input sample.
        """
        tiled = np.tile(input, reps=(self.thresholds.size, 1)).transpose()
        output = np.sum(tiled >= self.thresholds, axis=1)
        return output

    @abstractmethod
    def quantize(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Quantizes the input signal.

        Parameters:
            input: The input signal $x$ to be quantized.

        Returns:
            output: The quantized signal $y$.
        """
        return self.levels[self.digitize(input)]
