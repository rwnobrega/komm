from fractions import Fraction
from functools import cache, cached_property
from itertools import product
from typing import Self

import numpy as np
import numpy.typing as npt

from .._util.validators import (
    validate_integer,
    validate_integer_array,
    validate_pmf,
    validate_positive_integer,
)
from ..types import Array1D
from .util import (
    Word,
    canonical_code,
    infer_block_size,
    is_prefix_free,
    is_uniquely_parsable,
    kraft_parameter,
    parse_fixed_length,
    parse_prefix_free,
)


class FixedToVariableCode:
    r"""
    General fixed-to-variable length code. A *fixed-to-variable length code* with *source alphabet* $\mathcal{X}$, *target alphabet* $\mathcal{Y}$, and *source block size* $k$ is defined by an *encoding mapping* $\Enc: \mathcal{X}^k \to \mathcal{Y}^+$, where the domain is the set of all $k$-tuples with entries in $\mathcal{X}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{Y}$. The elements in the image of $\Enc$ are called *codewords*.

    A fixed-to-variable length code with $k = 1$ is equivalent to a [symbol code](/ref/SymbolCode).

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.
    """

    def __init__(
        self,
        source_cardinality: int,
        target_cardinality: int,
        source_block_size: int,
        enc_mapping: dict[Word, Word],
    ) -> None:
        calX = validate_integer(source_cardinality, low=2)
        calY = validate_integer(target_cardinality, low=2)
        k = validate_positive_integer(source_block_size)
        domain, codomain = enc_mapping.keys(), enc_mapping.values()
        if not set(domain) == set(product(range(calX), repeat=k)):
            raise ValueError("'enc_mapping': invalid domain")
        if not all(len(word) > 0 for word in codomain):
            raise ValueError("'enc_mapping': codewords must be non-empty")
        if not all(all(0 <= y < calY for y in word) for word in codomain):
            raise ValueError("'enc_mapping': invalid co-domain")
        self._source_cardinality = calX
        self._target_cardinality = calY
        self._source_block_size = k
        self._enc_mapping = dict(sorted(enc_mapping.items()))

    def __repr__(self) -> str:
        args = ", ".join([
            f"source_cardinality={self.source_cardinality}",
            f"target_cardinality={self.target_cardinality}",
            f"source_block_size={self.source_block_size}",
            f"enc_mapping={self.enc_mapping}",
        ])
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def from_enc_mapping(cls, enc_mapping: dict[Word, Word]) -> Self:
        r"""
        Constructs a fixed-to-variable length code from the encoding mapping $\Enc$.

        Parameters:
            enc_mapping: The encoding mapping $\Enc$. Must be a dictionary whose keys are all the $k$-tuples of integers in $\mathcal{X}$ and whose values are non-empty tuples of integers in $\mathcal{Y}$.

        Notes:
            The source block size $k$ is inferred from the domain of the encoding mapping, and the source and target cardinalities $|\mathcal{X}|$ and $|\mathcal{Y}|$ are inferred from the maximum values in the domain and co-domain, respectively.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0,): (0,),
            ...     (1,): (1, 0),
            ...     (2,): (1, 1),
            ... })
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]

            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1, 1),
            ...     (1, 0): (1, 1, 0),
            ...     (1, 1): (1, 0, 1),
            ... })
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (2, 2, 2)
            >>> code.codewords
            [(0,), (1, 1), (1, 1, 0), (1, 0, 1)]
        """
        domain, codomain = enc_mapping.keys(), enc_mapping.values()
        calX = max(max(word) for word in domain) + 1
        calY = max((y for word in codomain for y in word), default=0) + 1
        k = len(next(iter(domain)))
        return cls(calX, calY, k, enc_mapping)

    @classmethod
    def from_codewords(
        cls,
        codewords: list[Word],
        *,
        source_cardinality: int | None = None,
    ) -> Self:
        r"""
        Constructs a fixed-to-variable length code from a list of codewords.

        Parameters:
            codewords: The codewords of the code. Must be a list of length $|\mathcal{X}|^k$ containing non-empty tuples of integers in $\mathcal{Y}$. The tuple in position $i$ must be equal to $\Enc(\mathbf{x})$, where $\mathbf{x}$ is the $i$-th element in the lexicographic ordering of $\mathcal{X}^k$.

            source_cardinality: The source cardinality $|\mathcal{X}|$. Must be an integer greater than or equal to $2$. The default value is `len(codewords)`, which corresponds to source block size $k = 1$.

        Notes:
            The source block size $k$ is inferred from the number of codewords: it is the unique integer such that $|\mathcal{X}|^k$ equals `len(codewords)`. The target cardinality $|\mathcal{Y}|$ is inferred from the maximum value in the codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.enc_mapping
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}

            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     codewords=[(0,), (1, 1), (1, 1, 0), (1, 0, 1)],
            ...     source_cardinality=2,
            ... )
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (2, 2, 2)
            >>> code.enc_mapping
            {(0, 0): (0,),
             (0, 1): (1, 1),
             (1, 0): (1, 1, 0),
             (1, 1): (1, 0, 1)}

            >>> komm.FixedToVariableCode.from_codewords(
            ...     codewords=[(0,), (1, 0), (1, 1)],
            ...     source_cardinality=2,
            ... )
            Traceback (most recent call last):
            ...
            ValueError: length of 'codewords' must be a power of source cardinality 2 (got 3)
        """
        if not all(len(codeword) > 0 for codeword in codewords):
            raise ValueError("elements of 'codewords' must be non-empty")
        if source_cardinality is None:
            source_cardinality = len(codewords)
        calX = validate_integer(source_cardinality, low=2)
        calY = max(max(codeword) for codeword in codewords) + 1
        k = infer_block_size(len(codewords), calX, "codewords")
        enc_mapping = dict(zip(product(range(calX), repeat=k), codewords))
        return cls(calX, calY, k, enc_mapping)

    @classmethod
    def from_lengths(
        cls,
        lengths: npt.ArrayLike,
        *,
        source_cardinality: int | None = None,
        target_cardinality: int = 2,
    ) -> Self:
        r"""
        Constructs a fixed-to-variable length code from a list of codeword lengths. The codewords are assigned in lexicographic order, which yields the *canonical code* with the given lengths. The resulting code is always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

        Parameters:
            lengths: The codeword lengths of the code. Must be a 1D-array of length $|\mathcal{X}|^k$ of positive integers satisfying the [Kraft inequality](/ref/FixedToVariableCode/#kraft_parameter). The integer in position $i$ must be equal to $\ell(\mathbf{x})$, the length of the codeword $\Enc(\mathbf{x})$, where $\mathbf{x}$ is the $i$-th element in the lexicographic ordering of $\mathcal{X}^k$.

            source_cardinality: The source cardinality $|\mathcal{X}|$. Must be an integer greater than or equal to $2$. The default value is `len(lengths)`, which corresponds to source block size $k = 1$.

            target_cardinality: The target cardinality $|\mathcal{Y}|$. Must be an integer greater than or equal to $2$. The default value is $2$ (binary code).

        Notes:
            The source block size $k$ is inferred from the number of lengths: it is the unique integer such that $|\mathcal{X}|^k$ equals `len(lengths)`.

            If the lengths are non-decreasing, the resulting code is moreover *alphabetic*: the codewords preserve the lexicographic ordering of the source words. See <cite>CT06, Sec. 5.7</cite>.

        Examples:
            >>> code = komm.FixedToVariableCode.from_lengths([1, 2, 2])
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.enc_mapping
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}

            >>> code = komm.FixedToVariableCode.from_lengths(
            ...     lengths=[1, 2, 3, 3],
            ...     source_cardinality=2,
            ... )
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (2, 2, 2)
            >>> code.enc_mapping
            {(0, 0): (0,),
             (0, 1): (1, 0),
             (1, 0): (1, 1, 0),
             (1, 1): (1, 1, 1)}

            >>> code = komm.FixedToVariableCode.from_lengths(
            ...     lengths=[1, 2, 2, 2],
            ...     target_cardinality=3,
            ... )
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (4, 3, 1)
            >>> code.enc_mapping
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1),
             (3,): (1, 2)}

            >>> komm.FixedToVariableCode.from_lengths([1, 1, 1])
            Traceback (most recent call last):
            ...
            ValueError: 'lengths' must satisfy Kraft inequality
        """
        lengths = validate_integer_array(lengths, low=1, ndim=1)
        if source_cardinality is None:
            source_cardinality = lengths.size
        calX = validate_integer(source_cardinality, low=2)
        calY = validate_integer(target_cardinality, low=2)
        k = infer_block_size(lengths.size, calX, "lengths")
        codewords = canonical_code(lengths, base=calY)
        enc_mapping = dict(zip(product(range(calX), repeat=k), codewords))
        return cls(calX, calY, k, enc_mapping)

    @property
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$ of the code. It is the number of symbols in the source alphabet.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.source_cardinality
            3
        """
        return self._source_cardinality

    @property
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $|\mathcal{Y}|$ of the code. It is the number of symbols in the target alphabet.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @property
    def source_block_size(self) -> int:
        r"""
        The source block size $k$ of the code. It is the number of symbols in each source word.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.source_block_size
            1
        """
        return self._source_block_size

    @cached_property
    def size(self) -> int:
        r"""
        The number of codewords in the code. It is equal to $|\mathcal{X}|^k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.size
            3
        """
        return len(self.enc_mapping)

    @property
    def enc_mapping(self) -> dict[Word, Word]:
        r"""
        The encoding mapping $\Enc$ of the code. It is a dictionary of length $|\mathcal{X}|^k$ whose keys are all the $k$-tuples of integers in $\mathcal{X}$ and whose values are the corresponding codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.enc_mapping
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}
        """
        return self._enc_mapping

    @cached_property
    def _inv_enc_mapping(self) -> dict[Word, Word]:
        return {y: x for x, y in self.enc_mapping.items()}

    @cached_property
    def codewords(self) -> list[Word]:
        r"""
        The codewords of the code. They correspond to the image of the encoding mapping $\Enc$: the tuple in position $i$ is equal to $\Enc(\mathbf{x})$, where $\mathbf{x}$ is the $i$-th element in the lexicographic ordering of $\mathcal{X}^k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0,): (0,),
            ...     (1,): (1, 0),
            ...     (2,): (1, 1),
            ... })
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]
        """
        return list(self.enc_mapping.values())

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        The codeword lengths of the code. The integer in position $i$ is equal to $\ell(\mathbf{x})$, the length of the codeword $\Enc(\mathbf{x})$, where $\mathbf{x}$ is the $i$-th element in the lexicographic ordering of $\mathcal{X}^k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.lengths
            [1, 2, 2]

            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1, 1),
            ...     (1, 0): (1, 0, 0),
            ...     (1, 1): (1, 0, 1),
            ... })
            >>> code.lengths
            [1, 2, 3, 3]
        """
        return [len(y) for y in self.codewords]

    @cache
    def is_uniquely_decodable(self) -> bool:
        r"""
        Returns whether the code is uniquely decodable. A code is *uniquely decodable* if there is a unique way to parse any concatenation of codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.is_uniquely_decodable()  # 010 can be parsed as 0|10 or 01|0
            False
        """
        return self.is_prefix_free() or is_uniquely_parsable(self.codewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no codeword is a prefix of any other codeword.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.is_prefix_free()
            False

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.is_prefix_free()
            False
        """
        return is_prefix_free(self.codewords)

    @cache
    def kraft_parameter(self) -> Fraction:
        r"""
        Computes the Kraft parameter $K$ of the code. This quantity is given by
        $$
            K = \sum_{\mathbf{x} \in \mathcal{X}^k} |\mathcal{Y}|^{-{\ell(\mathbf{x})}},
        $$
        where $\ell(\mathbf{x})$ is the length of the codeword $\Enc(\mathbf{x})$ associated with the source word $\mathbf{x} \in \mathcal{X}^k$.

        Returns:
            kraft_parameter: The Kraft parameter $K$ of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     codewords=[(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1)],
            ... )
            >>> code.kraft_parameter()
            Fraction(3, 4)

            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     codewords=[(0,), (1, 0), (1, 1, 0), (1, 1, 1)],
            ... )
            >>> code.kraft_parameter()
            Fraction(1, 1)

            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     codewords=[(0, 0), (1, 1), (0,), (1,)],
            ... )
            >>> code.kraft_parameter()
            Fraction(3, 2)
        """
        return kraft_parameter(self.lengths, self.target_cardinality)

    @cache
    def is_complete(self) -> bool:
        r"""
        Returns whether the code is complete. A code is *complete* if it is uniquely decodable and its Kraft parameter is equal to $1$. See <cite>MacK03, Sec. 5.2</cite>.

        A prefix-free code is complete if and only if every infinite target sequence has a prefix that is a codeword, as in [variable-to-fixed codes](/ref/VariableToFixedCode/#is_complete).

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.is_complete()
            True

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 1)])
            >>> code.is_complete()
            True

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0, 1), (1, 0)])
            >>> code.is_complete()  # Not uniquely decodable
            False

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0)])
            >>> code.is_complete()  # Kraft parameter is 3/4
            False
        """
        return self.kraft_parameter() == 1 and self.is_uniquely_decodable()

    def rate(self, pmf: npt.ArrayLike) -> float:
        r"""
        Computes the expected rate $R$ of the code, considering a given (first-order) pmf $p$ over $\mathcal{X}$. This quantity is given by
        $$
            R = \frac{\bar{n}}{k},
        $$
        where $\bar{n}$ is the average codeword length, assuming iid source symbols drawn according to $p$. It is measured in $|\mathcal{Y}|$-ary digits per source symbol.

        Parameters:
            pmf: The pmf $p$ to be considered. Must be a 1D-array of size $|\mathcal{X}|$ of non-negative values that sum to $1$.

        Returns:
            rate: The expected rate $R$ of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
            >>> code.rate([1/2, 1/4, 1/4])
            1.5
        """
        k = self.source_block_size
        pmf = validate_pmf(pmf, size=self.source_cardinality)
        lengths = self.lengths
        probabilities = [np.prod(ps) for ps in product(pmf, repeat=k)]
        rate = np.dot(lengths, probabilities) / k
        return float(rate)

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code, which must be uniquely decodable.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$ and have a length that is a multiple of $k$.

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1, 1),
            ...     (1, 0): (1, 0, 0),
            ...     (1, 1): (1, 0, 1),
            ... })

            >>> code.encode([0, 1, 0, 0])
            array([1, 1, 0])

            >>> code.encode([0, 1, 0])  # Not a multiple of the source block size
            Traceback (most recent call last):
            ...
            ValueError: length of input must be a multiple of block size 2 (got 3)

            >>> code.encode([0, 7, 0, 0])  # 7 is not a source symbol
            Traceback (most recent call last):
            ...
            ValueError: elements of 'input' must be in [0:2)

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0,1), (1,0)])
            >>> code.encode([0, 1, 0])  # Code is not uniquely decodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely decodable
        """
        if not self.is_uniquely_decodable():
            raise ValueError("code is not uniquely decodable")
        calX = self.source_cardinality
        input = validate_integer_array(input, low=0, high=calX, ndim=1)
        return parse_fixed_length(input, self.enc_mapping, self.source_block_size)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code, which must be uniquely decodable.

        Warning:
            Decoding for non-prefix-free codes is not implemented yet.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a concatenation of codewords (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $\mathcal{X}$ with a length that is a multiple of $k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1, 1),
            ...     (1, 0): (1, 0, 0),
            ...     (1, 1): (1, 0, 1),
            ... })

            >>> code.decode([1, 1, 0])
            array([0, 1, 0, 0])

            >>> code.decode([0, 0, 1])  # Not a concatenation of codewords
            Traceback (most recent call last):
            ...
            ValueError: input contains an incomplete codeword

            >>> code = komm.FixedToVariableCode.from_codewords([(0,), (0,1), (1,0)])
            >>> code.decode([0, 1, 0])  # Code is not uniquely decodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely decodable
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
        )
