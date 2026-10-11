from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ..types import Array1D
from .util import Word

T = TypeVar("T")


class SymbolCode(ABC):
    r"""
    Abstract base class for symbol codes, which assign to each source symbol a non-empty codeword over the target alphabet.
    """

    @property
    @abstractmethod
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$ of the code. It is the number of symbols in the source alphabet.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $|\mathcal{Y}|$ of the code. It is the number of symbols in the target alphabet.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def enc_mapping(self) -> dict[int, Word]:
        r"""
        The encoding mapping $\Enc$ of the code. It is a dictionary of length $|\mathcal{X}|$ whose keys are all the integers in $\mathcal{X}$ and whose values are the corresponding codewords.
        """
        raise NotImplementedError

    @cached_property
    def codewords(self) -> list[Word]:
        r"""
        The codewords of the code. They correspond to the image of the encoding mapping $\Enc$: the tuple in position $x$ is equal to $\Enc(x)$.
        """
        return [self.enc_mapping[x] for x in range(self.source_cardinality)]

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        The codeword lengths of the code. The integer in position $x$ is equal to $\ell(x)$, the length of the codeword $\Enc(x)$.
        """
        return [len(y) for y in self.codewords]


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
