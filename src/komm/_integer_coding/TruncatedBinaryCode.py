from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index

from .. import abc
from .base import from_binary, take, to_binary


@dataclass
class TruncatedBinaryCode(abc.IntegerCode):
    r"""
    Truncated binary code. It is an integer code with domain $[0 : M)$, where $M \geq 2$ is a given cardinality. Let $k = \lfloor \log_2 M \rfloor$ and $u = 2^{k+1} - M$. The codeword for an integer $n \in [0 : M)$ is the $k$-bit binary representation of $n$, if $n < u$, or the $(k + 1)$-bit binary representation of $n + u$, otherwise. If $M$ is a power of $2$, the code reduces to the fixed-length binary code. For more details, see [Wikipedia: Truncated binary encoding](https://en.wikipedia.org/wiki/Truncated_binary_encoding).

    Parameters:
        cardinality: The cardinality $M$ of the code. Must satisfy $M \geq 2$.
    """

    cardinality: int

    def __post_init__(self) -> None:
        self.cardinality = index(self.cardinality)
        if not self.cardinality >= 2:
            raise ValueError("'cardinality' must be at least 2")
        self._k = self.cardinality.bit_length() - 1
        self._u = 2 ** (self._k + 1) - self.cardinality

    def _validate(self, integer: int) -> None:
        if not 0 <= integer < self.cardinality:
            raise ValueError("input contains an out-of-range entry")

    def encode_single(self, integer: int) -> list[int]:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> code.encode_single(2)
            [1, 0]
            >>> code.encode_single(3)
            [1, 1, 0]
        """
        self._validate(integer)
        u, k = self._u, self._k
        if integer < u:
            return to_binary(integer, width=k)
        return to_binary(integer + u, width=k + 1)

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> bits = iter([1, 1, 0, 1, 0])
            >>> code.decode_single(bits)
            3
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 0]
        """
        u, k = self._u, self._k
        integer = from_binary(take(bits, k))
        if integer < u:
            return integer
        return 2 * integer + from_binary(take(bits, 1)) - u

    def length(self, integer: int) -> int:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> code.length(2), code.length(3)
            (2, 3)
        """
        u, k = self._u, self._k
        self._validate(integer)
        return k if integer < u else k + 1

    def encode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> list(code.encode([4, 1, 3]))
            [1, 1, 1, 0, 1, 1, 1, 0]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> list(code.decode([1, 1, 1, 0, 1, 1, 1, 0]))
            [4, 1, 3]
        """
        return super().decode(input)
