from abc import ABC, abstractmethod
from collections.abc import Iterator
from fractions import Fraction
from functools import cache, cached_property
from itertools import islice
from operator import index
from typing import Generic, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer, validate_integer_array, validate_pmf
from ..types import Array1D
from .util import (
    Word,
    is_complete,
    is_prefix_free,
    is_prefix_of,
    is_uniquely_parsable,
    kraft_parameter,
    parse_fixed_length,
    parse_prefix_free,
    parse_uniquely_decodable,
)

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

    @cached_property
    def _max_length(self) -> int:
        return max(self.lengths)

    @cache
    def is_uniquely_decodable(self) -> bool:
        r"""
        Returns whether the code is uniquely decodable. A code is *uniquely decodable* if there is a unique way to parse any concatenation of codewords.
        """
        return self.is_prefix_free() or is_uniquely_parsable(self.codewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no codeword is a prefix of any other codeword.
        """
        return is_prefix_free(self.codewords)

    @cache
    def kraft_parameter(self) -> Fraction:
        r"""
        Computes the Kraft parameter $K$ of the code. This quantity is given by
        $$
            K = \sum_{x \in \mathcal{X}} |\mathcal{Y}|^{-\ell(x)},
        $$
        where $\ell(x)$ is the length of the codeword $\Enc(x)$ associated with the source symbol $x \in \mathcal{X}$.

        Returns:
            kraft_parameter: The Kraft parameter $K$ of the code.
        """
        return kraft_parameter(self.lengths, self.target_cardinality)

    @cache
    def is_complete(self) -> bool:
        r"""
        Returns whether the code is complete. A code is *complete* if it is uniquely decodable and its Kraft parameter is equal to $1$. See <cite>MacK03, Sec. 5.2</cite>.

        A prefix-free code is complete if and only if every infinite target sequence has a prefix that is a codeword, as in [variable-to-fixed codes](/ref/VariableToFixedCode/#is_complete).
        """
        return self.kraft_parameter() == 1 and self.is_uniquely_decodable()

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

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a concatenation of codewords (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $\mathcal{X}$.
        """
        if not self.is_uniquely_decodable():
            raise ValueError("code is not uniquely decodable")
        calY = self.target_cardinality
        input = validate_integer_array(input, low=0, high=calY, ndim=1)
        if self.is_prefix_free():  # Faster parser
            return parse_prefix_free(
                input,
                dictionary=self._inv_enc_mapping,
                allow_incomplete=False,
            )
        return parse_uniquely_decodable(input, dictionary=self._inv_enc_mapping)

    def encode_single(self, symbol: SupportsIndex) -> Word:
        r"""
        Encodes a single source symbol into its codeword.

        Parameters:
            symbol: The source symbol to be encoded. Must be in $\mathcal{X}$.

        Returns:
            codeword: The codeword of the source symbol. It is a tuple with elements in $\mathcal{Y}$.
        """
        symbol = validate_integer(symbol, low=0, high=self.source_cardinality)
        return self.enc_mapping[symbol]

    def decode_single(self, symbols: Iterator[SupportsIndex]) -> int:
        r"""
        Decodes a single codeword from an iterator of target symbols, using the code, which must be prefix-free. This method consumes exactly the symbols of one codeword, leaving the iterator at the boundary with the next one; any remaining symbols are left untouched.

        Parameters:
            symbols: An iterator of target symbols starting with a complete codeword. It must be an iterator (as returned by `iter`), not a general iterable, since advancing it is part of the contract.

        Returns:
            symbol: The decoded source symbol.

        Notes:
            A ValueError is raised if the iterator is exhausted mid-codeword or if an invalid word is found.
        """
        if not self.is_prefix_free():
            raise ValueError("code is not prefix-free")
        word: Word = ()
        for y in islice(symbols, self._max_length):
            word += (index(y),)
            if word in self._inv_enc_mapping:
                return self._inv_enc_mapping[word][0]
        if any(is_prefix_of(word, codeword) for codeword in self.codewords):
            raise ValueError("input contains an incomplete codeword")
        raise ValueError("input contains invalid word")


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


class VariableToFixedCode(ABC):
    r"""
    Abstract base class for variable-to-fixed length codes, which parse the source sequence into sourcewords and represent each sourceword by a target word of fixed length.
    """

    @property
    @abstractmethod
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $|\mathcal{Y}|$ of the code. It is the number of symbols in the target alphabet.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$ of the code. It is the number of symbols in the source alphabet.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_block_size(self) -> int:
        r"""
        The target block size $n$ of the code. It is the number of symbols in each target block.
        """
        raise NotImplementedError

    @cached_property
    def size(self) -> int:
        r"""
        The number of sourcewords in the code. It is less than or equal to $|\mathcal{Y}|^n$.
        """
        return len(self.dec_mapping)

    @property
    @abstractmethod
    def dec_mapping(self) -> dict[Word, Word]:
        r"""
        The decoding mapping $\Dec$ of the code. It is a dictionary of length at most $|\mathcal{Y}|^n$ whose keys are $n$-tuples of integers in $\mathcal{Y}$ and whose values are the corresponding sourcewords.
        """
        raise NotImplementedError

    @cached_property
    def _inv_dec_mapping(self) -> dict[Word, Word]:
        return {x: y for y, x in self.dec_mapping.items()}

    @cached_property
    def sourcewords(self) -> list[Word]:
        r"""
        The sourcewords of the code. They correspond to the image of the decoding mapping $\Dec$: the tuple in position $i$ is equal to $\Dec(\mathbf{y})$, where $\mathbf{y}$ is the $i$-th element in the lexicographic ordering of the domain of $\Dec$.
        """
        return list(self.dec_mapping.values())

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        The sourceword lengths of the code. The integer in position $i$ is the length of the $i$-th sourceword.
        """
        return [len(x) for x in self.sourcewords]

    @cache
    def is_complete(self) -> bool:
        """
        Returns whether the code is complete. A code is *complete* if every infinite source sequence has a prefix that is a sourceword. See <cite>JS72, Definition 2, p. 766</cite>.
        """
        return is_complete(self.sourcewords, self.source_cardinality)

    @cache
    def is_uniquely_encodable(self) -> bool:
        r"""
        Returns whether the code is uniquely encodable. A code is *uniquely encodable* if there is a unique way to parse any concatenation of sourcewords.
        """
        return self.is_prefix_free() or is_uniquely_parsable(self.sourcewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no sourceword is a prefix of any other sourceword.
        """
        return is_prefix_free(self.sourcewords)

    def rate(self, pmf: npt.ArrayLike) -> float:
        r"""
        Computes the expected rate $R$ of the code, considering a given (first-order) pmf $p$ over $\mathcal{X}$. This quantity is given by
        $$
            R = \frac{n}{\bar{k}},
        $$
        where $\bar{k}$ is the average sourceword length, assuming iid source symbols drawn according to $p$. It is measured in $|\mathcal{Y}|$-ary digits per source symbol.

        Parameters:
            pmf: The pmf $p$ to be considered. Must be a 1D-array of size $|\mathcal{X}|$ of non-negative values that sum to $1$.

        Returns:
            rate: The expected rate $R$ of the code.
        """
        n = self.target_block_size
        pmf = validate_pmf(pmf, size=self.source_cardinality)
        lengths = self.lengths
        probabilities = [np.prod([pmf[symb] for symb in x]) for x in self.sourcewords]
        rate = n / np.dot(lengths, probabilities)
        return float(rate)

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code, which must be complete and uniquely encodable. When the input sequence ends with symbols that form only a partial match with any sourceword, the encoder will complete this last block using any valid sourceword that starts with these remaining symbols.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $\mathcal{Y}$ with a length that is a multiple of $n$.
        """
        if not self.is_complete():
            raise ValueError("code is not complete")
        if not self.is_uniquely_encodable():
            raise ValueError("code is not uniquely encodable")
        calX = self.source_cardinality
        input = validate_integer_array(input, low=0, high=calX, ndim=1)
        # Complete and uniquely encodable implies prefix-free
        return parse_prefix_free(
            input,
            dictionary=self._inv_dec_mapping,
            allow_incomplete=True,
        )

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code, which must be complete and uniquely encodable.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$ and have a length that is a multiple of $n$. Also, the sequence must be a concatenation of target words (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $\mathcal{X}$.
        """
        if not self.is_complete():
            raise ValueError("code is not complete")
        if not self.is_uniquely_encodable():
            raise ValueError("code is not uniquely encodable")
        calY = self.target_cardinality
        input = validate_integer_array(input, low=0, high=calY, ndim=1)
        return parse_fixed_length(input, self.dec_mapping, self.target_block_size)
