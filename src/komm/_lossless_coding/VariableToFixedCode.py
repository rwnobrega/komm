from functools import cache, cached_property
from itertools import count, product
from typing import Self

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.validators import validate_integer, validate_positive_integer
from ..types import Array1D
from .util import Word


class VariableToFixedCode(abc.VariableToFixedCode):
    r"""
    General variable-to-fixed length code. A *variable-to-fixed length code* with *target alphabet* $\mathcal{Y}$, *source alphabet* $\mathcal{X}$, and *target block size* $n$ is defined by a (possibly partial) decoding mapping $\Dec: \mathcal{Y}^n \rightharpoonup \mathcal{X}^+$, where the domain is the set of all $n$-tuples with entries in $\mathcal{Y}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{X}$. The elements in the image of $\Dec$ are called *sourcewords*.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.
    """

    def __init__(
        self,
        target_cardinality: int,
        source_cardinality: int,
        target_block_size: int,
        dec_mapping: dict[Word, Word],
    ) -> None:
        calY = validate_integer(target_cardinality, low=2)
        calX = validate_integer(source_cardinality, low=2)
        n = validate_positive_integer(target_block_size)
        domain, codomain = dec_mapping.keys(), dec_mapping.values()
        if not set(domain) <= set(product(range(calY), repeat=n)):
            raise ValueError("'dec_mapping': invalid domain")
        if not all(len(word) > 0 for word in codomain):
            raise ValueError("'dec_mapping': sourcewords must be non-empty")
        if not all(all(0 <= x < calX for x in word) for word in codomain):
            raise ValueError("'dec_mapping': invalid co-domain")
        self._target_cardinality = calY
        self._source_cardinality = calX
        self._target_block_size = n
        self._dec_mapping = dict(sorted(dec_mapping.items()))

    def __repr__(self) -> str:
        args = ", ".join([
            f"target_cardinality={self.target_cardinality}",
            f"source_cardinality={self.source_cardinality}",
            f"target_block_size={self.target_block_size}",
            f"dec_mapping={self.dec_mapping}",
        ])
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def from_dec_mapping(cls, dec_mapping: dict[Word, Word]) -> Self:
        r"""
        Constructs a variable-to-fixed length code from the decoding mapping $\Dec$.

        Parameters:
            dec_mapping: The decoding mapping $\Dec$. Must be a dictionary of length at most $|\mathcal{Y}|^n$ whose keys are $n$-tuples of integers in $\mathcal{Y}$ and whose values are non-empty tuples of integers in $\mathcal{X}$.

        Notes:
            The target block size $n$ is inferred from the domain of the decoding mapping, and the target and source cardinalities $|\mathcal{Y}|$ and $|\mathcal{X}|$ are inferred from the maximum values in the domain and co-domain, respectively.

        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0): (0, 0, 0),
            ...     (0, 1): (0, 0, 1),
            ...     (1, 0): (0, 1),
            ...     (1, 1): (1,),
            ... })
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (2, 2, 2)
            >>> code.sourcewords
            [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]

            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0, 0): (1,),
            ...     (0, 0, 1): (2,),
            ...     (0, 1, 0): (0, 1),
            ...     (0, 1, 1): (0, 2),
            ...     (1, 0, 0): (0, 0, 0),
            ...     (1, 0, 1): (0, 0, 1),
            ...     (1, 1, 0): (0, 0, 2),
            ... })  # Incomplete mapping
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (2, 3, 3)
            >>> code.sourcewords
            [(1,), (2,), (0, 1), (0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)]
        """
        domain, codomain = dec_mapping.keys(), dec_mapping.values()
        calY = max(max(word) for word in domain) + 1
        calX = max((x for word in codomain for x in word), default=0) + 1
        n = len(next(iter(domain)))
        return cls(calY, calX, n, dec_mapping)

    @classmethod
    def from_sourcewords(
        cls,
        sourcewords: list[Word],
        *,
        target_cardinality: int = 2,
    ) -> Self:
        r"""
        Constructs a variable-to-fixed length code from a list of sourcewords.

        Parameters:
            sourcewords: The sourcewords of the code. Must be a list of length at most $|\mathcal{Y}|^n$ containing non-empty tuples of integers in $\mathcal{X}$. The tuple in position $i$ must be equal to $\Dec(\mathbf{y})$, where $\mathbf{y}$ is the $i$-th element in the lexicographic ordering of $\mathcal{Y}^n$.

            target_cardinality: The target cardinality $|\mathcal{Y}|$. Must be an integer greater than or equal to $2$. The default value is $2$ (binary code).

        Note:
            The target block size $n$ is inferred from the number of sourcewords, and the source cardinality $|\mathcal{X}|$ is inferred from the maximum value in the sourcewords.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(
            ...     sourcewords=[(0, 0, 0), (0, 0, 1), (0, 1), (1,)],
            ... )
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (2, 2, 2)
            >>> code.dec_mapping
            {(0, 0): (0, 0, 0),
             (0, 1): (0, 0, 1),
             (1, 0): (0, 1),
             (1, 1): (1,)}

            >>> code = komm.VariableToFixedCode.from_sourcewords(
            ...     sourcewords=[
            ...         (1,), (2,), (0, 1), (0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)
            ...     ],
            ... )
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (2, 3, 3)
            >>> code.dec_mapping
            {(0, 0, 0): (1,),
             (0, 0, 1): (2,),
             (0, 1, 0): (0, 1),
             (0, 1, 1): (0, 2),
             (1, 0, 0): (0, 0, 0),
             (1, 0, 1): (0, 0, 1),
             (1, 1, 0): (0, 0, 2)}

            >>> code = komm.VariableToFixedCode.from_sourcewords(
            ...     sourcewords=[(0, 0), (0, 1), (1,)],
            ...     target_cardinality=3,
            ... )
            >>> code.target_cardinality, code.source_cardinality, code.target_block_size
            (3, 2, 1)
            >>> code.dec_mapping
            {(0,): (0, 0), (1,): (0, 1), (2,): (1,)}
        """
        if not all(len(sourceword) > 0 for sourceword in sourcewords):
            raise ValueError("elements of 'sourcewords' must be non-empty")
        calY = validate_integer(target_cardinality, low=2)
        calX = max(max(word) for word in sourcewords) + 1
        n = next(n for n in count(1) if calY**n >= len(sourcewords))
        dec_mapping = dict(zip(product(range(calY), repeat=n), sourcewords))
        return cls(calY, calX, n, dec_mapping)

    @property
    def target_cardinality(self) -> int:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @property
    def source_cardinality(self) -> int:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.source_cardinality
            2
        """
        return self._source_cardinality

    @property
    def target_block_size(self) -> int:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.target_block_size
            2
        """
        return self._target_block_size

    @cached_property
    def size(self) -> int:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.size
            3
        """
        return super().size

    @property
    def dec_mapping(self) -> dict[Word, Word]:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.dec_mapping
            {(0, 0): (0,), (0, 1): (1,), (1, 0): (0, 1)}
        """
        return self._dec_mapping

    @cached_property
    def sourcewords(self) -> list[Word]:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1,),
            ...     (1, 0): (0, 1),
            ... })
            >>> code.sourcewords
            [(0,), (1,), (0, 1)]
        """
        return super().sourcewords

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.lengths
            [1, 1, 2]
        """
        return super().lengths

    @cache
    def is_fully_covering(self) -> bool:
        """
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1, 0), (1, 1)])
            >>> code.is_fully_covering()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.is_fully_covering()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0, 0), (0, 1)])
            >>> code.is_fully_covering()  # (1,) is not covered
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (0, 1)])
            >>> code.is_fully_covering()  # (1,) is not covered
            False
        """
        return super().is_fully_covering()

    @cache
    def is_uniquely_encodable(self) -> bool:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1, 0), (1, 1)])
            >>> code.is_uniquely_encodable()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.is_uniquely_encodable()  # 01 can be parsed as 0|1 or 01
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0, 0), (0, 1)])
            >>> code.is_uniquely_encodable()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (0, 1)])
            >>> code.is_uniquely_encodable()
            True
        """
        return super().is_uniquely_encodable()

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1, 0), (1, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.is_prefix_free()
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0, 0), (0, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (0, 1)])
            >>> code.is_prefix_free()
            False
        """
        return super().is_prefix_free()

    def rate(self, pmf: npt.ArrayLike) -> float:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.rate([2/3, 1/3])
            1.3846153846153846
        """
        return super().rate(pmf)

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0): (0, 0, 0),
            ...     (0, 1): (0, 0, 1),
            ...     (1, 0): (0, 1),
            ...     (1, 1): (1,),
            ... })

            >>> code.encode([1, 0, 0, 0])  # Parsed as 1|000
            array([1, 1, 0, 0])

            >>> code.encode([1, 0, 0])  # Incomplete input, completed as 1|000
            array([1, 1, 0, 0])

            >>> code.encode([0, 7, 0, 0])  # 7 is not a source symbol
            Traceback (most recent call last):
            ...
            ValueError: elements of 'input' must be in [0:2)

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0, 0), (0, 1)])
            >>> code.encode([1, 0, 0, 0])  # Code is not fully covering
            Traceback (most recent call last):
            ...
            ValueError: code is not fully covering

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.encode([1, 0, 0, 0])  # Code is not uniquely encodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely encodable
        """
        return super().encode(input)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1,),
            ...     (1, 0): (2,),
            ... })
            >>> code.decode([0, 0, 1, 0])
            array([0, 2])

            >>> code.decode([1, 1, 0, 0, 1])  # Not a multiple of target block size
            Traceback (most recent call last):
            ...
            ValueError: length of input must be a multiple of block size 2 (got 5)

            >>> code.decode([0, 0, 1, 1])  # 11 is not a valid target word
            Traceback (most recent call last):
            ...
            ValueError: input contains invalid word

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0, 0), (0, 1)])
            >>> code.decode([0, 0, 0, 1])  # Code is not fully covering
            Traceback (most recent call last):
            ...
            ValueError: code is not fully covering

            >>> code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
            >>> code.decode([0, 0, 0, 1])  # Code is not uniquely encodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely encodable
        """
        return super().decode(input)
