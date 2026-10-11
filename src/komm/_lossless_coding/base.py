from abc import ABC, abstractmethod
from functools import cache, cached_property
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer_array, validate_pmf
from ..types import Array1D
from .util import Word, is_prefix_free, is_uniquely_parsable, parse_prefix_free

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
    def _inv_enc_mapping(self) -> dict[Word, Word]:
        return {y: (x,) for x, y in self.enc_mapping.items()}

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

    @cache
    def is_uniquely_decodable(self) -> bool:
        r"""
        Returns whether the code is uniquely decodable. A code is *uniquely decodable* if there is a unique way to parse any concatenation of codewords.
        """
        return is_uniquely_parsable(self.codewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no codeword is a prefix of any other codeword.
        """
        return is_prefix_free(self.codewords)

    @cache
    def kraft_parameter(self) -> float:
        r"""
        Computes the Kraft parameter $K$ of the code. This quantity is given by
        $$
            K = \sum_{x \in \mathcal{X}} |\mathcal{Y}|^{-\ell(x)},
        $$
        where $\ell(x)$ is the length of the codeword $\Enc(x)$ associated with the source symbol $x \in \mathcal{X}$.

        Returns:
            kraft_parameter: The Kraft parameter $K$ of the code.
        """
        calY = self.target_cardinality
        kraft_parameter = np.sum(np.float_power(calY, -np.asarray(self.lengths)))
        return float(kraft_parameter)

    def average_length(self, pmf: npt.ArrayLike) -> float:
        r"""
        Computes the average codeword length $\bar{n}$ of the code, considering a given pmf $p$ over $\mathcal{X}$. This quantity is given by
        $$
            \bar{n} = \sum_{x \in \mathcal{X}} p(x) \ell(x),
        $$
        where $\ell(x)$ is the length of the codeword $\Enc(x)$. It is measured in $|\mathcal{Y}|$-ary digits per source symbol.

        Parameters:
            pmf: The pmf $p$ to be considered. Must be a 1D-array of size $|\mathcal{X}|$ of non-negative values that sum to $1$.

        Returns:
            average_length: The average codeword length $\bar{n}$ of the code.
        """
        pmf = validate_pmf(pmf, size=self.source_cardinality)
        return float(np.dot(self.lengths, pmf))

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code, which must be uniquely decodable.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $\mathcal{Y}$.
        """
        if not self.is_uniquely_decodable():
            raise ValueError("code is not uniquely decodable")
        calX = self.source_cardinality
        input = validate_integer_array(input, low=0, high=calX, ndim=1)
        output = [y for x in input.tolist() for y in self.enc_mapping[x]]
        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code, which must be uniquely decodable.

        Warning:
            Decoding for non-prefix-free codes is not implemented yet.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a concatenation of codewords (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $\mathcal{X}$.
        """
        if not self.is_uniquely_decodable():
            raise ValueError("code is not uniquely decodable")
        if not self.is_prefix_free():
            raise NotImplementedError(
                "decoding for non-prefix-free codes is not implemented yet"
            )
        calY = self.target_cardinality
        input = validate_integer_array(input, low=0, high=calY, ndim=1)
        return parse_prefix_free(
            input,
            dictionary=self._inv_enc_mapping,
            allow_incomplete=False,
            cardinality=calY,
        )


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
