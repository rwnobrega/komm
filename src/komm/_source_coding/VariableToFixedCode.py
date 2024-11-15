import itertools as it
from typing import cast

import numpy as np
import numpy.typing as npt
from attrs import field, frozen, validators
from typing_extensions import Self

from .._validation import is_pmf, validate_call
from .util import Word, is_prefix_free


@frozen
class VariableToFixedCode:
    r"""
    Variable-to-fixed length code. A *variable-to-fixed length code* with *target alphabet* $\mathcal{T}$, *source alphabet* $\mathcal{S}$, and *target block size* $n$ is defined by a (possibly partial) injective decoding mapping $\mathrm{Dec} : \mathcal{T}^n \to \mathcal{S}^+$, where the domain is the set of all $n$-tuples with entries in $\mathcal{T}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{S}$. Here, we assume that $\mathcal{T} = [0:T)$ and $\mathcal{S} = [0:S)$, for integers $T \geq 2$ and $S \geq 2$. The elements in the image of $\mathrm{Dec}$ are called *sourcewords*.

    Attributes:
        target_cardinality: The target cardinality $T$.
        source_cardinality: The source cardinality $S$.
        target_block_size: The target block size $n$.
        dec_mapping: The decoding mapping $\mathrm{Dec}$ of the code. Must be a dictionary of length at most $S^n$ whose keys are $n$-tuples of integers in $[0:T)$ and whose values are distinct non-empty tuples of integers in $[0:S)$.
    """

    target_cardinality: int = field(validator=validators.ge(2))
    source_cardinality: int = field(validator=validators.ge(2))
    target_block_size: int = field(validator=validators.ge(1))
    dec_mapping: dict[Word, Word] = field()

    def __attrs_post_init__(self) -> None:
        domain, codomain = self.dec_mapping.keys(), self.dec_mapping.values()
        T, S, n = (
            self.target_cardinality,
            self.source_cardinality,
            self.target_block_size,
        )
        if not set(domain) <= set(it.product(range(T), repeat=n)):
            raise ValueError(f"'dec_mapping': invalid domain")
        if not all(
            all(0 <= x < S for x in word) and len(word) > 0 for word in codomain
        ):
            raise ValueError(f"'dec_mapping': invalid co-domain")
        if len(set(codomain)) != len(codomain):
            raise ValueError(f"'dec_mapping': non-injective mapping")

    @classmethod
    def from_dec_mapping(cls, dec_mapping: dict[Word, Word]) -> Self:
        r"""
        Constructs a variable-to-fixed code from the decoding map $\Dec$.

        Parameters:
            dec_mapping: The decoding map $\Dec$. See the corresponding attribute for more details.

        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({(0,0): (0,0,0), (0,1): (0,0,1), (1,0): (0,1), (1,1): (1,)})
            >>> (code.target_cardinality, code.source_cardinality, code.target_block_size)
            (2, 2, 2)
            >>> code.dec_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0): (0, 0, 0),
             (0, 1): (0, 0, 1),
             (1, 0): (0, 1),
             (1, 1): (1,)}

            >>> code = komm.VariableToFixedCode.from_dec_mapping({(0,0,0): (1,), (0,0,1): (2,), (0,1,0): (0,1), (0,1,1): (0,2), (1,0,0): (0,0,0), (1,0,1): (0,0,1), (1,1,0): (0,0,2)})
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (2, 3, 3)
            >>> code.dec_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0, 0): (1,),
             (0, 0, 1): (2,),
             (0, 1, 0): (0, 1),
             (0, 1, 1): (0, 2),
             (1, 0, 0): (0, 0, 0),
             (1, 0, 1): (0, 0, 1),
             (1, 1, 0): (0, 0, 2)}
        """
        domain, codomain = dec_mapping.keys(), dec_mapping.values()
        T = max(max(word) for word in domain) + 1
        S = max(max(word) for word in codomain) + 1
        n = len(next(iter(domain)))
        return cls(T, S, n, dec_mapping)

    @classmethod
    @validate_call(target_cardinality=field(validator=validators.ge(2)))
    def from_sourcewords(cls, target_cardinality: int, sourcewords: list[Word]) -> Self:
        r"""
        Constructs a variable-to-fixed code from the target cardinality $T$ and a list of sourcewords.

        Parameters:
            target_cardinality: The target cardinality $T$. Must be an integer greater than or equal to $2$.
            sourcewords: The sourcewords of the code. See the [corresponding property](./#sourcewords) for more details.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
            >>> (code.target_cardinality, code.source_cardinality, code.target_block_size)
            (2, 2, 2)
            >>> code.dec_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0): (0, 0, 0),
             (0, 1): (0, 0, 1),
             (1, 0): (0, 1),
             (1, 1): (1,)}
        """
        T = target_cardinality
        S = max(max(word) for word in sourcewords) + 1
        n = next(n for n in it.count(1) if T ** n >= len(sourcewords))
        dec_mapping = dict(zip(it.product(range(T), repeat=n), sourcewords))
        return cls(T, S, n, dec_mapping)

    @property
    def sourcewords(self) -> list[Word]:
        r"""
        The sourcewords of the code. It is a list of length at most $T^n$ containing tuples of integers in $[0:S)$. The tuple in position $i$ of `sourcewords` is equal to $\mathrm{Dec}(v)$, where $v$ is the $i$-th element in the lexicographic ordering of $[0:T)^n$.

        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({(0,0): (0,0,0), (0,1): (0,0,1), (1,0): (0,1), (1,1): (1,)})
            >>> code.sourcewords
            [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
        """
        return list(self.dec_mapping.values())

    @property
    def inv_dec_mapping(self) -> dict[Word, Word]:
        r"""
        The inverse decoding mapping $\mathrm{Dec}^{-1}$ of the code. It is a dictionary of length at most $T^n$ whose keys are all the sourcewords of the code, and whose values are the corresponding target words.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.inv_dec_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0, 0): (0, 0),
             (0, 0, 1): (0, 1),
             (0, 1): (1, 0),
             (1,): (1, 1)}
        """
        return {v: k for k, v in self.dec_mapping.items()}

    def is_unique_encodable(self) -> bool:
        r"""
        Returns whether the code is unique encodable or not. [Not implemented yet].
        """
        raise NotImplementedError

    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free or not. A code is *prefix-free* if no sourceword is a prefix of any other sourceword.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,1,0), (0,1), (1,)])
            >>> code.is_prefix_free()
            False
        """
        return is_prefix_free(self.sourcewords)

    @validate_call(pmf=field(converter=np.asarray, validator=is_pmf))
    def rate(self, pmf: npt.ArrayLike) -> float:
        r"""
        Computes the expected rate $R$ of the code, considering a given pmf. This quantity is given by
        $$
            R = \frac{n}{\bar{k}},
        $$
        where $n$ is the target block size, and $\bar{k}$ is the expected sourceword length, assuming iid source symbols drawn from $p_X$. It is measured in $T$-ary digits per source symbol.

        Parameters:
            pmf (Array1D[float]): The (first-order) probability mass function $p_X$ to be considered.

        Returns:
            rate: The expected rate $R$ of the code.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.rate([2/3, 1/3])
            np.float64(0.9473684210526315)
        """
        pmf = cast(npt.NDArray[np.float64], pmf)
        probabilities = [np.prod([pmf[x] for x in word]) for word in self.sourcewords]
        lengths = [len(word) for word in self.sourcewords]
        return self.target_block_size / np.dot(lengths, probabilities)
