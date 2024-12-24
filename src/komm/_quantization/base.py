from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ScalarQuantizer(ABC):
    @property
    @abstractmethod
    def levels(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer levels $v_0, v_1, \ldots, v_{L-1}$.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def thresholds(self) -> npt.NDArray[np.floating]:
        r"""
        The quantizer finite thresholds $t_1, t_2, \ldots, t_{L-1}$.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Quantizes the input signal.

        Parameters:
            input: The input signal $x$ to be quantized.

        Returns:
            output: The quantized signal $y$.
        """
        tiled = np.tile(input, reps=(self.thresholds.size, 1)).transpose()
        output = self.levels[np.sum(tiled >= self.thresholds, axis=1)]
        return output
