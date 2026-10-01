from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

from .. import abc
from .base import validate_positive


@dataclass
class UnaryCode(abc.IntegerCode):
    r"""
    Unary code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of $n - 1$ copies of one bit followed by a single copy of the other bit, called the *stop bit*. For more details, see [Wikipedia: Unary coding](https://en.wikipedia.org/wiki/Unary_coding) or <cite>MacK03, Ch. 7</cite>.

    Parameters:
        stop_bit: The stop bit. Must be either $0$ or $1$. The default value is $1$, so that the codeword for $n$ consists of $n - 1$ zeros followed by a single $1$.
    """

    stop_bit: int = 1

    def __post_init__(self) -> None:
        self.stop_bit = index(self.stop_bit)
        if self.stop_bit not in {0, 1}:
            raise ValueError("'stop_bit' must be either 0 or 1")

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.encode_single(4)
            [0, 0, 0, 1]

            >>> code = komm.UnaryCode(stop_bit=0)
            >>> code.encode_single(4)
            [1, 1, 1, 0]
        """
        integer = validate_positive(integer)
        return [1 - self.stop_bit] * (integer - 1) + [self.stop_bit]

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> bits = iter([0, 0, 0, 1, 1, 0])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 0]

            >>> code = komm.UnaryCode(stop_bit=0)
            >>> bits = iter([1, 1, 1, 0, 0, 1])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [0, 1]
        """
        for pos, bit in enumerate(bits):
            if bit == self.stop_bit:
                return pos + 1
            if bit != 1 - self.stop_bit:
                raise ValueError(f"invalid bit in input: {bit}")
        raise ValueError("input contains an incomplete codeword")

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.length(4)
            4
        """
        return validate_positive(integer)

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> list(code.encode([4, 1, 3]))
            [0, 0, 0, 1, 1, 0, 0, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> list(code.decode([0, 0, 0, 1, 1, 0, 0, 1]))
            [4, 1, 3]
        """
        return super().decode(input)
