from functools import cache, cached_property
from itertools import count, product

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .._util.information_theory import PMF
from .util import (
    Word,
    is_prefix_free,
    is_uniquely_parsable,
    parse_fixed_length,
    parse_prefix_free,
)


class FixedToVariableCode:
    r"""
    General fixed-to-variable length code. A *fixed-to-variable length code* with *source alphabet* $\mathcal{S}$, *target alphabet* $\mathcal{T}$, and *source block size* $k$ is defined by an *encoding mapping* $\Enc : \mathcal{S}^k \to \mathcal{T}^+$, where the domain is the set of all $k$-tuples with entries in $\mathcal{S}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{T}$. Here we assume that $\mathcal{S} = [0:S)$ and $\mathcal{T} = [0:T)$, for integers $S \geq 2$ and $T \geq 2$. The elements in the image of $\Enc$ are called *codewords*.
    """

    def __init__(
        self,
        source_cardinality: int,
        target_cardinality: int,
        source_block_size: int,
        enc_mapping: dict[Word, Word],
    ) -> None:
        self._source_cardinality = source_cardinality
        self._target_cardinality = target_cardinality
        self._source_block_size = source_block_size
        self._enc_mapping = enc_mapping
        self.__post_init__()

    def __post_init__(self) -> None:
        domain, codomain = self.enc_mapping.keys(), self.enc_mapping.values()
        S = self.source_cardinality
        T = self.target_cardinality
        k = self.source_block_size
        if not S >= 2:
            raise ValueError("'source_cardinality': must be at least 2")
        if not T >= 2:
            raise ValueError("'target_cardinality': must be at least 2")
        if not k >= 1:
            raise ValueError("'source_block_size': must be at least 1")
        if set(domain) != set(product(range(S), repeat=k)):
            raise ValueError(f"'enc_mapping': invalid domain")
        if not all(all(0 <= x < T for x in word) for word in codomain):
            raise ValueError(f"'enc_mapping': invalid co-domain")

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
            enc_mapping: The encoding mapping $\Enc$. Must be a dictionary whose keys are all the $k$-tuples of integers in $[0:S)$ and whose values are non-empty tuples of integers in $[0:T)$.

        Notes:
            The source block size $k$ is inferred from the domain of the encoding mapping, and the source and target cardinalities $S$ and $T$ are inferred from the maximum values in the domain and co-domain, respectively.

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
        S = max(max(word) for word in domain) + 1
        T = max(max(word) for word in codomain) + 1
        k = len(next(iter(domain)))
        return cls(S, T, k, enc_mapping)

    @classmethod
    def from_codewords(cls, source_cardinality: int, codewords: list[Word]) -> Self:
        r"""
        Constructs a fixed-to-variable length code from the source cardinality $S$ and a list of codewords.

        Parameters:
            source_cardinality: The source cardinality $S$. Must be an integer greater than or equal to $2$.

            codewords: The codewords of the code. Must be a list of length $S^k$ containing tuples of integers in $[0:T)$, where $T$ is the target cardinality of the code. The tuple in position $i$ must be equal to $\Enc(u)$, where $u$ is the $i$-th element in the lexicographic ordering of $[0:S)^k$.

        Notes:
            The source block size $k$ is inferred from the length of the codewords, and the target cardinality $T$ is inferred from the maximum value in the codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     source_cardinality=3,
            ...     codewords=[(0,), (1,0), (1,1)],
            ... )
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.enc_mapping
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}

            >>> code = komm.FixedToVariableCode.from_codewords(
            ...     source_cardinality=2,
            ...     codewords=[(0,), (1,1), (1,1,0), (1,0,1)]
            ... )
            >>> (code.source_cardinality, code.target_cardinality, code.source_block_size)
            (2, 2, 2)
            >>> code.enc_mapping
            {(0, 0): (0,),
             (0, 1): (1, 1),
             (1, 0): (1, 1, 0),
             (1, 1): (1, 0, 1)}
        """
        S = source_cardinality
        T = max(max(codeword) for codeword in codewords) + 1
        k = next(k for k in count(1) if S**k >= len(codewords))
        enc_mapping = dict(zip(product(range(S), repeat=k), codewords))
        return cls(S, T, k, enc_mapping)

    @cached_property
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $S$ of the code. It is the number of symbols in the source alphabet.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.source_cardinality
            3
        """
        return self._source_cardinality

    @cached_property
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $T$ of the code. It is the number of symbols in the target alphabet.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @cached_property
    def source_block_size(self) -> int:
        r"""
        The source block size $k$ of the code. It is the number of symbols in each source block.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.source_block_size
            1
        """
        return self._source_block_size

    @cached_property
    def size(self) -> int:
        r"""
        The number of codewords in the code. It is equal to $S^k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.size
            3
        """
        return len(self.enc_mapping)

    @cached_property
    def enc_mapping(self) -> dict[Word, Word]:
        r"""
        The encoding mapping $\Enc$ of the code. It is a dictionary of length $S^k$ whose keys are all the $k$-tuples of integers in $[0:S)$ and whose values are the corresponding codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.enc_mapping
            {(0,): (0,), (1,): (1, 0), (2,): (1, 1)}
        """
        return self._enc_mapping

    @cached_property
    def codewords(self) -> list[Word]:
        r"""
        The codewords of the code. They correspond to the image of the encoding mapping $\Enc$.

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
    def _inv_enc_mapping(self) -> dict[Word, Word]:
        return {v: k for k, v in self.enc_mapping.items()}

    @cache
    def is_uniquely_decodable(self) -> bool:
        r"""
        Returns whether the code is uniquely decodable. A code is *uniquely decodable* if there is a unique way to parse any concatenation of codewords.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,0)])
            >>> code.is_uniquely_decodable()  # 010 can be parsed as 0|10 or 01|0
            False
        """
        return is_uniquely_parsable(self.codewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no codeword is a prefix of any other codeword.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,1)])
            >>> code.is_prefix_free()
            False

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,0)])
            >>> code.is_prefix_free()
            False
        """
        return is_prefix_free(self.codewords)

    @cache
    def kraft_parameter(self) -> float:
        r"""
        Computes the Kraft parameter $K$ of the code. This quantity is given by
        $$
            K = \sum_{u \in \mathcal{S}^k} T^{-{\ell_u}},
        $$
        where $\ell_u$ is the length of the codeword $\Enc(u)$, $T$ is the target cardinality, and $k$ is the source block size.

        Returns:
            kraft_parameter: The Kraft parameter $K$ of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(5, [(0,0,0), (0,0,1), (0,1,0), (1,0,1), (1,1)])
            >>> code.kraft_parameter()
            np.float64(0.75)

            >>> code = komm.FixedToVariableCode.from_codewords(4, [(0,), (1,0), (1,1,0), (1,1,1)])
            >>> code.kraft_parameter()
            np.float64(1.0)

            >>> code = komm.FixedToVariableCode.from_codewords(4, [(0,0), (1,1), (0,), (1,)])
            >>> code.kraft_parameter()
            np.float64(1.5)
        """
        T = self.target_cardinality
        lengths = np.array([len(word) for word in self.codewords])
        return np.sum(np.float_power(T, -lengths))

    def rate(self, pmf: npt.ArrayLike) -> float:
        r"""
        Computes the expected rate $R$ of the code, considering a given pmf. This quantity is given by
        $$
            R = \frac{\bar{n}}{k},
        $$
        where $\bar{n}$ is the expected codeword length, assuming iid source symbols drawn from $p_X$, and $k$ is the source block size. It is measured in $T$-ary digits per source symbol.

        Parameters:
            pmf: The (first-order) probability mass function $p_X$ to be considered.

        Returns:
            rate: The expected rate $R$ of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.rate([1/2, 1/4, 1/4])
            np.float64(1.5)
        """
        pmf = PMF(pmf)
        k = self.source_block_size
        probabilities = [np.prod(ps) for ps in product(pmf, repeat=k)]
        lengths = [len(word) for word in self.codewords]
        return np.dot(lengths, probabilities) / k

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code, which must be uniquely decodable.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code) and have a length that is a multiple of the source block size $k$.

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code).

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

            >>> code.encode([0, 7, 0, 0])  # 07 is not a valid source word
            Traceback (most recent call last):
            ...
            ValueError: input contains invalid word

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,0)])
            >>> code.encode([0, 1, 0])  # Code is not uniquely decodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely decodable
        """
        if not self.is_uniquely_decodable():
            raise ValueError("code is not uniquely decodable")
        input = np.asarray(input)
        return parse_fixed_length(input, self.enc_mapping, self.source_block_size)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code, which must be uniquely decodable.

        Warning:
            Decoding for non-prefix-free codes is not implemented yet.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code). Also, the sequence must be a concatenation of codewords (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code) with a length that is a multiple of the source block size $k$.

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
            ValueError: input contains invalid word

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,0)])
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
        input = np.asarray(input)
        return parse_prefix_free(input, self._inv_enc_mapping, allow_incomplete=False)
