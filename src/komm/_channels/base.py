from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.information_theory import LogBase


class DiscreteMemorylessChannel(ABC):
    @cached_property
    @abstractmethod
    def input_cardinality(self) -> int:
        r"""
        The channel input cardinality $|\mathcal{X}|$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def output_cardinality(self) -> int:
        r"""
        The channel output cardinality $|\mathcal{Y}|$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        r"""
        The channel transition probability matrix $p_{Y \mid X}$.
        """
        raise NotImplementedError

    @abstractmethod
    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""Returns the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel.

        Parameters:
            input_pmf: The probability mass function $p_X$ of the channel input $X$. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

            base: The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

        Returns:
            The mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$.
        """
        raise NotImplementedError

    @abstractmethod
    def capacity(self, base: LogBase = 2.0) -> float:
        r"""
        Returns the channel capacity $C$.

        Parameters:
            base: The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

        Returns:
            The channel capacity $C$.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Transmits the input sequence through the channel and returns the output sequence.

        Parameters:
            input: The input sequence.

        Returns:
            output: The output sequence.
        """
        raise NotImplementedError
