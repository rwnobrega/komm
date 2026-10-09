from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ..types import Array1D

T = TypeVar("T")


class TokenCode(ABC, Generic[T]):
    r"""
    Abstract base class for codes which parse the source sequence into tokens and represent each token in the target alphabet.
    """

    source_cardinality: int
    target_cardinality: int

    @abstractmethod
    def source_to_tokens(self, source: npt.ArrayLike) -> list[T]:
        r"""
        Encodes a given sequence of source symbols to the corresponding list of tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def tokens_to_source(self, tokens: list[T]) -> Array1D[np.integer]:
        r"""
        Decodes a given list of tokens to the corresponding sequence of source symbols.
        """
        raise NotImplementedError

    @abstractmethod
    def tokens_to_target(self, tokens: list[T]) -> Array1D[np.integer]:
        r"""
        Returns the target alphabet representation corresponding to a given list of tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def target_to_tokens(self, target: npt.ArrayLike) -> list[T]:
        r"""
        Returns the list of tokens corresponding to a given target alphabet representation.
        """
        raise NotImplementedError

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.
        """
        tokens = self.source_to_tokens(input)
        output = self.tokens_to_target(tokens)
        return output

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of target symbols to a sequence of source symbols.

        Parameters:
            input: The sequence of target symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded source symbols. It is a 1D-array with elements in $\mathcal{X}$.
        """
        tokens = self.target_to_tokens(input)
        output = self.tokens_to_source(tokens)
        return output
