import itertools as it

import numpy as np
import numpy.typing as npt
from attrs import frozen
from typing_extensions import Self

from .._util.information_theory import PMF
from .util import Word, is_prefix_free, is_uniquely_decodable, parse_prefix_free


@frozen
class FixedToVariableCode:
    r"""
    Fixed-to-variable length code. A *fixed-to-variable length code* with *source alphabet* $\mathcal{S}$, *target alphabet* $\mathcal{T}$, and *source block size* $k$ is defined by an injective *encoding mapping* $\Enc : \mathcal{S}^k \to \mathcal{T}^+$, where the domain is the set of all $k$-tuples with entries in $\mathcal{S}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{T}$. Here we assume that $\mathcal{S} = [0:S)$ and $\mathcal{T} = [0:T)$, for integers $S \geq 2$ and $T \geq 2$. The elements in the image of $\Enc$ are called *codewords*.

    Attributes:
        source_cardinality: The source cardinality $S$.
        target_cardinality: The target cardinality $T$.
        source_block_size: The source block size $k$.
        enc_mapping: The encoding mapping $\Enc$ of the code. Must be a dictionary of length $S^k$ whose keys are $k$-tuples of integers in $[0:S)$ and whose values are distinct non-empty tuples of integers in $[0:T)$.
    """

    source_cardinality: int
    target_cardinality: int
    source_block_size: int
    enc_mapping: dict[Word, Word]

    def __attrs_post_init__(self) -> None:
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
        if set(domain) != set(it.product(range(S), repeat=k)):
            raise ValueError(f"'enc_mapping': invalid domain")
        if not all(
            all(0 <= x < T for x in word) and len(word) > 0 for word in codomain
        ):
            raise ValueError(f"'enc_mapping': invalid co-domain")
        if len(set(codomain)) != len(codomain):
            raise ValueError(f"'enc_mapping': non-injective mapping")

    @classmethod
    def from_enc_mapping(cls, enc_mapping: dict[Word, Word]) -> Self:
        r"""
        Constructs a fixed-to-variable length code from the encoding mapping $\Enc$.

        Parameters:
            enc_mapping: The encoding mapping $\Enc$. See the corresponding attribute for more details.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({(0,): (0,), (1,): (1,0), (2,): (1,1)})
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}

            >>> code = komm.FixedToVariableCode.from_enc_mapping({(0,0): (0,), (0,1): (1,0,0), (1,0): (1,1), (1,1): (1,0,1)})
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (2, 2, 2)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0): (0,),
             (0, 1): (1, 0, 0),
             (1, 0): (1, 1),
             (1, 1): (1, 0, 1)}
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
            codewords: The codewords of the code. See the [corresponding property](./#codewords) for more details.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.source_cardinality, code.target_cardinality, code.source_block_size
            (3, 2, 1)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0,): (0,),
             (1,): (1, 0),
             (2,): (1, 1)}

            >>> code = komm.FixedToVariableCode.from_codewords(2, [(0,), (1,0,0), (1,1), (1,0,1)])
            >>> (code.source_cardinality, code.target_cardinality, code.source_block_size)
            (2, 2, 2)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0): (0,),
             (0, 1): (1, 0, 0),
             (1, 0): (1, 1),
             (1, 1): (1, 0, 1)}
        """
        S = source_cardinality
        T = max(max(codeword) for codeword in codewords) + 1
        k = next(k for k in it.count(1) if S**k >= len(codewords))
        enc_mapping = dict(zip(it.product(range(S), repeat=k), codewords))
        return cls(S, T, k, enc_mapping)

    @property
    def codewords(self) -> list[Word]:
        r"""
        The codewords of the code. It is a list of length $S^k$ containing tuples of integers in $[0:T)$. The tuple in position $i$ of `codewords` is equal to $\Enc(u)$, where $u$ is the $i$-th element in the lexicographic ordering of $[0:S)^k$.

        Examples:
            >>> code = komm.FixedToVariableCode.from_enc_mapping({(0,): (0,), (1,): (1,0), (2,): (1,1)})
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]
        """
        return list(self.enc_mapping.values())

    @property
    def inv_enc_mapping(self) -> dict[Word, Word]:
        r"""
        The inverse encoding mapping $\Enc^{-1}$ of the code. It is a dictionary of length $S^k$ whose keys are all the codewords of the code and whose values are the corresponding source words.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.inv_enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0,): (0,),
             (1, 0): (1,),
             (1, 1): (2,)}
        """
        return {v: k for k, v in self.enc_mapping.items()}

    def is_uniquely_decodable(self) -> bool:
        r"""
        Returns whether the code is uniquely decodable or not. A code is *uniquely decodable* if
        $$
            s_1 \cdots s_n \neq s'_1 \cdots s'_m \implies \Enc(s_1) \cdots \Enc(s_n) \neq \Enc(s'_1) \cdots \Enc(s'_m).
        $$

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,1)])
            >>> code.is_uniquely_decodable()
            True

            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (0,1), (1,0)])
            >>> code.is_uniquely_decodable()
            False
        """
        return is_uniquely_decodable(self.codewords)

    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free or not. A code is *prefix-free* if no codeword is a prefix of any other codeword.

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
            >>> code.rate([0.5, 0.25, 0.25])
            np.float64(1.5)
        """
        pmf = PMF(pmf)
        k = self.source_block_size
        probabilities = [np.prod(ps) for ps in it.product(pmf, repeat=k)]
        lengths = [len(word) for word in self.codewords]
        return np.dot(lengths, probabilities) / k

    def encode(self, source_symbols: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code.

        Parameters:
            source_symbols: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

        Returns:
            The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.encode([1, 0, 1, 0, 2, 0])
            array([1, 0, 0, 1, 0, 0, 1, 1, 0])
        """
        source_symbols = np.asarray(source_symbols)
        k, enc = self.source_block_size, self.enc_mapping
        return np.concatenate([enc[tuple(s)] for s in source_symbols.reshape(-1, k)])

    def decode(self, target_symbols: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code. Only works if the code is prefix-free.

        Parameters:
            target_symbols: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> code.decode([1, 0, 0, 1, 0, 0, 1, 1, 0])
            array([1, 0, 1, 0, 2, 0])

            >>> code = komm.FixedToVariableCode.from_codewords(2, [(0,), (1,0), (1,1), (1,1,0)])
            >>> code.decode([1, 0, 0, 1, 0, 0, 1, 1, 0])
            Traceback (most recent call last):
            ...
            ValueError: code is not prefix-free
        """
        if not self.is_prefix_free():
            raise ValueError("code is not prefix-free")
        target_symbols = np.asarray(target_symbols)
        return parse_prefix_free(target_symbols, self.inv_enc_mapping)
