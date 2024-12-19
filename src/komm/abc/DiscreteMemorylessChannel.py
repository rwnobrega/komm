from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .._util.information_theory import LogBase


class DiscreteMemorylessChannel(ABC):
    @property
    @abstractmethod
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        pass

    @property
    @abstractmethod
    def input_cardinality(self) -> int:
        pass

    @property
    @abstractmethod
    def output_cardinality(self) -> int:
        pass

    @abstractmethod
    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        pass

    @abstractmethod
    def capacity(self, base: LogBase = 2.0) -> float:
        pass

    @abstractmethod
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Transmits the input sequence through the channel and returns the output sequence.

        Parameters:
            input: The input sequence.

        Returns:
            output: The output sequence.
        """
        pass
