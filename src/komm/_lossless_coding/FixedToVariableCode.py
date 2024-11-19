import itertools as it
from typing import cast

import numpy as np
import numpy.typing as npt
from attrs import field, frozen, validators
from typing_extensions import Self

from .._validation import is_pmf, validate_call
from .util import Word, is_prefix_free, is_uniquely_decodable


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

    source_cardinality: int = field(validator=validators.ge(2))
    target_cardinality: int = field(validator=validators.ge(2))
    source_block_size: int = field(validator=validators.ge(1))
    enc_mapping: dict[Word, Word] = field()

    def __attrs_post_init__(self) -> None:
        domain, codomain = self.enc_mapping.keys(), self.enc_mapping.values()
        S, T, k = (
            self.source_cardinality,
            self.target_cardinality,
            self.source_block_size,
        )
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
    @validate_call(source_cardinality=field(validator=validators.ge(2)))
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

    @validate_call(pmf=field(converter=np.asarray, validator=is_pmf))
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
        pmf = cast(npt.NDArray[np.float64], pmf)
        k = self.source_block_size
        probabilities = [np.prod(ps) for ps in it.product(pmf, repeat=k)]
        lengths = [len(word) for word in self.codewords]
        return np.dot(lengths, probabilities) / k
