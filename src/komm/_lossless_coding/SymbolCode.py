from collections.abc import Iterator
from functools import cache, cached_property
from typing import Self, SupportsIndex

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.validators import validate_integer, validate_integer_array
from ..types import Array1D
from .util import Word, canonical_code


class SymbolCode(abc.SymbolCode):
    r"""
    General symbol code. A *symbol code* with *source alphabet* $\mathcal{X}$ and *target alphabet* $\mathcal{Y}$ is defined by an *encoding mapping* $\Enc: \mathcal{X} \to \mathcal{Y}^+$, where the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{Y}$. The elements in the image of $\Enc$ are called *codewords*. For more details, see <cite>MacK03, Ch. 5</cite>.

    A symbol code is equivalent to a [fixed-to-variable length code](/ref/FixedToVariableCode) with source block size $k = 1$, in which each source symbol $x$ corresponds to the $1$-tuple $(x)$.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.
    """

    def __init__(
        self,
        source_cardinality: int,
        target_cardinality: int,
        enc_mapping: dict[int, Word],
    ) -> None:
        calX = validate_integer(source_cardinality, low=2)
        calY = validate_integer(target_cardinality, low=2)
        domain, codomain = enc_mapping.keys(), enc_mapping.values()
        if not set(domain) == set(range(calX)):
            raise ValueError("'enc_mapping': invalid domain")
        if not all(len(word) > 0 for word in codomain):
            raise ValueError("'enc_mapping': codewords must be non-empty")
        if not all(all(0 <= y < calY for y in word) for word in codomain):
            raise ValueError("'enc_mapping': invalid co-domain")
        self._source_cardinality = calX
        self._target_cardinality = calY
        self._enc_mapping = {x: enc_mapping[x] for x in range(calX)}

    def __repr__(self) -> str:
        args = ", ".join([
            f"source_cardinality={self.source_cardinality}",
            f"target_cardinality={self.target_cardinality}",
            f"enc_mapping={self.enc_mapping}",
        ])
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def from_enc_mapping(cls, enc_mapping: dict[int, Word]) -> Self:
        r"""
        Constructs a symbol code from the encoding mapping $\Enc$.

        Parameters:
            enc_mapping: The encoding mapping $\Enc$. Must be a dictionary whose keys are all the integers in $\mathcal{X}$ and whose values are non-empty tuples of integers in $\mathcal{Y}$.

        Notes:
            The source cardinality $|\mathcal{X}|$ is inferred from the size of the domain of the encoding mapping, and the target cardinality $|\mathcal{Y}|$ from the maximum value in its co-domain.

        Examples:
            >>> code = komm.SymbolCode.from_enc_mapping({
            ...     0: (0,),
            ...     1: (1, 0),
            ...     2: (1, 1),
            ... })
            >>> code.source_cardinality, code.target_cardinality
            (3, 2)
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]
        """
        calY = max((y for word in enc_mapping.values() for y in word), default=0) + 1
        return cls(len(enc_mapping), calY, enc_mapping)

    @classmethod
    def from_codewords(cls, codewords: list[Word]) -> Self:
        r"""
        Constructs a symbol code from a list of codewords.

        Parameters:
            codewords: The codewords of the code. Must be a list of length $|\mathcal{X}|$ containing non-empty tuples of integers in $\mathcal{Y}$. The tuple in position $x$ must be equal to $\Enc(x)$.

        Notes:
            The source cardinality $|\mathcal{X}|$ is inferred from the number of codewords, and the target cardinality $|\mathcal{Y}|$ from the maximum value in the codewords.

        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.source_cardinality, code.target_cardinality
            (3, 2)
            >>> code.enc_mapping
            {0: (0,),
             1: (1, 0),
             2: (1, 1)}

            >>> code = komm.SymbolCode.from_codewords([(0,), (1,), (2, 0), (2, 1)])
            >>> code.source_cardinality, code.target_cardinality
            (4, 3)
        """
        if not all(len(word) > 0 for word in codewords):
            raise ValueError("elements of 'codewords' must be non-empty")
        calY = max((y for word in codewords for y in word), default=0) + 1
        return cls(len(codewords), calY, dict(enumerate(codewords)))

    @classmethod
    def from_lengths(
        cls,
        lengths: npt.ArrayLike,
        *,
        target_cardinality: int = 2,
    ) -> Self:
        r"""
        Constructs a symbol code from a list of codeword lengths. The codewords are assigned in lexicographic order, which yields the *canonical code* with the given lengths. The resulting code is always [prefix-free](/ref/SymbolCode/#is_prefix_free) (hence [uniquely decodable](/ref/SymbolCode/#is_uniquely_decodable)).

        Parameters:
            lengths: The codeword lengths of the code. Must be a 1D-array of length $|\mathcal{X}|$ of positive integers satisfying the [Kraft inequality](/ref/SymbolCode/#kraft_parameter). The integer in position $x$ must be equal to $\ell(x)$, the length of the codeword $\Enc(x)$.

            target_cardinality: The target cardinality $|\mathcal{Y}|$. Must be an integer greater than or equal to $2$. The default value is $2$ (binary code).

        Notes:
            The source cardinality $|\mathcal{X}|$ is inferred from the number of lengths.

            If the lengths are non-decreasing, the resulting code is moreover *alphabetic*: the codewords preserve the order of the source symbols. See <cite>CT06, Sec. 5.7</cite>.

        Examples:
            >>> code = komm.SymbolCode.from_lengths([1, 2, 2])
            >>> code.enc_mapping
            {0: (0,),
             1: (1, 0),
             2: (1, 1)}

            >>> code = komm.SymbolCode.from_lengths([1, 2, 2, 2], target_cardinality=3)
            >>> code.enc_mapping
            {0: (0,),
             1: (1, 0),
             2: (1, 1),
             3: (1, 2)}

            >>> komm.SymbolCode.from_lengths([1, 1, 1])
            Traceback (most recent call last):
            ...
            ValueError: 'lengths' must satisfy Kraft inequality
        """
        lengths = validate_integer_array(lengths, low=1, ndim=1)
        calY = validate_integer(target_cardinality, low=2)
        codewords = canonical_code(lengths, base=calY)
        return cls(lengths.size, calY, dict(enumerate(codewords)))

    @property
    def source_cardinality(self) -> int:
        r"""
        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.source_cardinality
            3
        """
        return self._source_cardinality

    @property
    def target_cardinality(self) -> int:
        r"""
        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @property
    def enc_mapping(self) -> dict[int, Word]:
        r"""
        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.enc_mapping
            {0: (0,),
             1: (1, 0),
             2: (1, 1)}
        """
        return self._enc_mapping

    @cached_property
    def codewords(self) -> list[Word]:
        r"""
        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]
        """
        return super().codewords

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.lengths
            [1, 2, 2]
        """
        return super().lengths

    @cache
    def is_uniquely_decodable(self) -> bool:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.is_uniquely_decodable()  # 010 can be parsed as 0|10 or 01|0
            False
        """
        return super().is_uniquely_decodable()

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.is_prefix_free()
            False

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.is_prefix_free()
            False
        """
        return super().is_prefix_free()

    @cache
    def kraft_parameter(self) -> float:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0)])
            >>> code.kraft_parameter()
            0.75

            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.kraft_parameter()
            1.0

            >>> code = komm.SymbolCode.from_codewords([(0, 0), (1, 1), (0,), (1,)])
            >>> code.kraft_parameter()
            1.5
        """
        return super().kraft_parameter()

    def average_length(self, pmf: npt.ArrayLike) -> float:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.average_length([1/2, 1/4, 1/4])
            1.5
        """
        return super().average_length(pmf)

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])

            >>> code.encode([0, 1, 0, 2])
            array([0, 1, 0, 0, 1, 1])

            >>> code.encode([0, 3, 0])  # 3 is not a source symbol
            Traceback (most recent call last):
            ...
            ValueError: elements of 'input' must be in [0:3)

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.encode([0, 1, 0])  # Code is not uniquely decodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely decodable
        """
        return super().encode(input)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])

            >>> code.decode([0, 1, 0, 0, 1, 1])
            array([0, 1, 0, 2])

            >>> code.decode([0, 1])  # Not a concatenation of codewords
            Traceback (most recent call last):
            ...
            ValueError: input contains invalid word

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.decode([0, 1, 0])  # Code is not uniquely decodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely decodable
        """
        return super().decode(input)

    def encode_single(self, symbol: SupportsIndex) -> Word:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])

            >>> code.encode_single(2)
            (1, 1)

            >>> code.encode_single(3)  # 3 is not a source symbol
            Traceback (most recent call last):
            ...
            ValueError: 'symbol' must be in [0:3) (got 3)
        """
        return super().encode_single(symbol)

    def decode_single(self, symbols: Iterator[SupportsIndex]) -> int:
        r"""
        Examples:
            >>> code = komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)])

            >>> symbols = iter([1, 1, 0, 1, 0])
            >>> code.decode_single(symbols)
            2
            >>> list(symbols)  # Iterator is left at codeword boundary
            [0, 1, 0]

            >>> code.decode_single(iter([1]))  # Incomplete codeword
            Traceback (most recent call last):
            ...
            ValueError: input contains an incomplete codeword

            >>> code = komm.SymbolCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.decode_single(iter([0, 1]))  # Code is not prefix-free
            Traceback (most recent call last):
            ...
            ValueError: code is not prefix-free
        """
        return super().decode_single(symbols)
