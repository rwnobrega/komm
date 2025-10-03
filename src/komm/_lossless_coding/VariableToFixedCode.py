from functools import cache, cached_property
from itertools import count, product

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .._util.validators import validate_pmf
from ..types import Array1D
from .util import (
    Word,
    is_fully_covering,
    is_prefix_free,
    is_uniquely_parsable,
    parse_fixed_length,
    parse_prefix_free,
)


class VariableToFixedCode:
    r"""
    General variable-to-fixed length code. A *variable-to-fixed length code* with *target alphabet* $\mathcal{Y}$, *source alphabet* $\mathcal{X}$, and *target block size* $n$ is defined by a (possibly partial) decoding mapping $\mathrm{Dec}: \mathcal{Y}^n \rightharpoonup \mathcal{X}^+$, where the domain is the set of all $n$-tuples with entries in $\mathcal{Y}$, and the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{X}$. The elements in the image of $\mathrm{Dec}$ are called *sourcewords*.

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
        self._target_cardinality = target_cardinality
        self._source_cardinality = source_cardinality
        self._target_block_size = target_block_size
        self._dec_mapping = dec_mapping
        self.__post_init__()

    def __post_init__(self) -> None:
        domain, codomain = self.dec_mapping.keys(), self.dec_mapping.values()
        calY, calX = self.target_cardinality, self.source_cardinality
        n = self.target_block_size
        if not calY >= 2:
            raise ValueError("'target_cardinality' must be at least 2")
        if not calX >= 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if not n >= 1:
            raise ValueError("'target_block_size' must be at least 1")
        if not set(domain) <= set(product(range(calY), repeat=n)):
            raise ValueError("'dec_mapping': invalid domain")
        if not all(
            all(0 <= x < calX for x in word) and len(word) > 0 for word in codomain
        ):
            raise ValueError("'dec_mapping': invalid co-domain")

    def __repr__(self) -> str:
        args = ", ".join([
            f"target_cardinality={self.target_cardinality}",
            f"source_cardinality={self.source_cardinality}",
            f"target_block_size={self.target_block_size}",
            f"dec_mapping={self.dec_mapping}",
        ])
        return f"{__class__.__name__}({args})"

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
            ...     (0, 0, 0): (1, ),
            ...     (0, 0, 1): (2, ),
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
        calX = max(max(word) for word in codomain) + 1
        n = len(next(iter(domain)))
        return cls(calY, calX, n, dec_mapping)

    @classmethod
    def from_sourcewords(cls, target_cardinality: int, sourcewords: list[Word]) -> Self:
        r"""
        Constructs a variable-to-fixed length code from the target cardinality $|\mathcal{Y}|$ and a list of sourcewords.

        Parameters:
            target_cardinality: The target cardinality $|\mathcal{Y}|$. Must be an integer greater than or equal to $2$.

            sourcewords: The sourcewords of the code. Must be a list of length at most $|\mathcal{Y}|^n$ containing tuples of integers in $\mathcal{X}$. The tuple in position $i$ must be equal to $\mathrm{Dec}(\mathbf{y})$, where $\mathbf{y}$ is the $i$-th element in the lexicographic ordering of $\mathcal{Y}^n$.

        Note:
            The target block size $n$ is inferred from the length of the sourcewords, and the source cardinality $|\mathcal{X}|$ is inferred from the maximum value in the sourcewords.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(
            ...     target_cardinality=2,
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
            ...     target_cardinality=2,
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
        """
        calY = target_cardinality
        calX = max(max(word) for word in sourcewords) + 1
        n = next(n for n in count(1) if calY**n >= len(sourcewords))
        dec_mapping = dict(zip(product(range(calY), repeat=n), sourcewords))
        return cls(calY, calX, n, dec_mapping)

    @cached_property
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $|\mathcal{Y}|$ of the code. It is the number of symbols in the target alphabet.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @cached_property
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$ of the code. It is the number of symbols in the source alphabet.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.source_cardinality
            2
        """
        return self._source_cardinality

    @cached_property
    def target_block_size(self) -> int:
        r"""
        The target block size $n$ of the code. It is the number of symbols in each target block.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.target_block_size
            2
        """
        return self._target_block_size

    @cached_property
    def size(self) -> int:
        r"""
        The number of sourcewords in the code. It is less than or equal to $|\mathcal{Y}|^n$.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.size
            3
        """
        return len(self.dec_mapping)

    @cached_property
    def dec_mapping(self) -> dict[Word, Word]:
        r"""
        The decoding mapping $\mathrm{Dec}$ of the code. It is a dictionary of length at most $|\mathcal{Y}|^n$ whose keys are $n$-tuples of integers in $\mathcal{Y}$ and whose values are the corresponding sourcewords.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.dec_mapping
            {(0, 0): (0,), (0, 1): (1,), (1, 0): (0, 1)}
        """
        return self._dec_mapping

    @cached_property
    def _inv_dec_mapping(self) -> dict[Word, Word]:
        return {x: y for y, x in self.dec_mapping.items()}

    @cached_property
    def sourcewords(self) -> list[Word]:
        r"""
        The sourcewords of the code. They correspond to the image of the decoding mapping $\mathrm{Dec}$.

        Examples:
            >>> code = komm.VariableToFixedCode.from_dec_mapping({
            ...     (0, 0): (0,),
            ...     (0, 1): (1,),
            ...     (1, 0): (0, 1),
            ... })
            >>> code.sourcewords
            [(0,), (1,), (0, 1)]
        """
        return list(self.dec_mapping.values())

    @cache
    def is_fully_covering(self) -> bool:
        """
        Returns whether the code is fully covering. A code is *fully covering* if every possible source sequence has a prefix that is a sourceword.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1, 0), (1, 1)])
            >>> code.is_fully_covering()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.is_fully_covering()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0, 0), (0, 1)])
            >>> code.is_fully_covering()  # (1,) is not covered
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (0, 1)])
            >>> code.is_fully_covering()  # (1,) is not covered
            False
        """
        return is_fully_covering(self.sourcewords, self.source_cardinality)

    @cache
    def is_uniquely_encodable(self) -> bool:
        r"""
        Returns whether the code is uniquely encodable. A code is *uniquely encodable* if there is a unique way to parse any concatenation of sourcewords.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1, 0), (1, 1)])
            >>> code.is_uniquely_encodable()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.is_uniquely_encodable()  # 01 can be parsed as 0|1 or 01
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0, 0), (0, 1)])
            >>> code.is_uniquely_encodable()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (0, 1)])
            >>> code.is_uniquely_encodable()
            True
        """
        return is_uniquely_parsable(self.sourcewords)

    @cache
    def is_prefix_free(self) -> bool:
        r"""
        Returns whether the code is prefix-free. A code is *prefix-free* if no sourceword is a prefix of any other sourceword.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1, 0), (1, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.is_prefix_free()
            False

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0, 0), (0, 1)])
            >>> code.is_prefix_free()
            True

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (0, 1)])
            >>> code.is_prefix_free()
            False
        """
        return is_prefix_free(self.sourcewords)

    def rate(self, pmf: npt.ArrayLike) -> np.floating:
        r"""
        Computes the expected rate $R$ of the code, considering a given (first-order) pmf $p$ over $\mathcal{X}$. This quantity is given by
        $$
            R = \frac{n}{\bar{k}},
        $$
        and $\bar{k}$ is the expected sourceword length, assuming iid source symbols drawn from $p$. It is measured in $|\mathcal{Y}|$-ary digits per source symbol.

        Parameters:
            pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        Returns:
            rate: The expected rate $R$ of the code.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.rate([2/3, 1/3])
            np.float64(1.3846153846153846)
        """
        n = self.target_block_size
        pmf = validate_pmf(pmf)
        probabilities = [np.prod([pmf[symb] for symb in x]) for x in self.sourcewords]
        lengths = [len(x) for x in self.sourcewords]
        return n / np.dot(lengths, probabilities)

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of source symbols using the code, which must be fully covering and uniquely encodable. When the input sequence ends with symbols that form only a partial match with any sourceword, the encoder will complete this last block using any valid sourceword that starts with these remaining symbols.

        Warning:
            Encoding for non-prefix-free codes is not implemented yet.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $\mathcal{Y}$ with a length that is a multiple of $n$.

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

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0, 0), (0, 1)])
            >>> code.encode([1, 0, 0, 0])  # Code is not fully covering
            Traceback (most recent call last):
            ...
            ValueError: code is not fully covering

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.encode([1, 0, 0, 0])  # Code is not uniquely encodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely encodable
        """
        if not self.is_fully_covering():
            raise ValueError("code is not fully covering")
        if not self.is_uniquely_encodable():
            raise ValueError("code is not uniquely encodable")
        if not self.is_prefix_free():
            raise NotImplementedError(
                "encoding for non-prefix-free codes is not implemented yet"
            )
        input = np.asarray(input)
        return parse_prefix_free(input, self._inv_dec_mapping, allow_incomplete=True)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of target symbols using the code, which must be fully covering and uniquely encodable.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$ and have a length that is a multiple of $n$. Also, the sequence must be a concatenation of target words (i.e., the output of the `encode` method).

        Returns:
            The sequence of decoded symbols. It is a 1D-array with elements in $\mathcal{X}$.

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

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0, 0), (0, 1)])
            >>> code.decode([0, 0, 0, 1])  # Code is not fully covering
            Traceback (most recent call last):
            ...
            ValueError: code is not fully covering

            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,), (1,), (0, 1)])
            >>> code.decode([0, 0, 0, 1])  # Code is not uniquely encodable
            Traceback (most recent call last):
            ...
            ValueError: code is not uniquely encodable
        """
        if not self.is_fully_covering():
            raise ValueError("code is not fully covering")
        if not self.is_uniquely_encodable():
            raise ValueError("code is not uniquely encodable")
        input = np.asarray(input)
        return parse_fixed_length(input, self.dec_mapping, self.target_block_size)
