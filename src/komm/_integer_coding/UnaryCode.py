from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, SupportsIndex

from .. import abc
from .._util.validators import (
    validate_choice,
    validate_integer,
    validate_positive_integer,
)


@dataclass(init=False)
class UnaryCode(abc.IntegerCode):
    r"""
    Unary code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of $n - 1$ copies of one bit followed by a single copy of the other bit, called the *stop bit*. For more details, see [Wikipedia: Unary coding](https://en.wikipedia.org/wiki/Unary_coding) or <cite>MacK03, Ch. 7</cite>.

    The table below shows the codewords for the first integers.

    | $n$ | Stop bit $1$ | Stop bit $0$ |
    | :-: | ------------ | ------------ |
    | $1$ | `1`          | `0`          |
    | $2$ | `01`         | `10`         |
    | $3$ | `001`        | `110`        |
    | $4$ | `0001`       | `1110`       |
    | $5$ | `00001`      | `11110`      |
    | $6$ | `000001`     | `111110`     |
    | $7$ | `0000001`    | `1111110`    |
    | $8$ | `00000001`   | `11111110`   |

    Parameters:
        stop_bit: The stop bit. Must be either $0$ or $1$. The default value is $1$, so that the codeword for $n$ consists of $n - 1$ zeros followed by a single $1$.
    """

    stop_bit: int

    def __init__(self, stop_bit: Literal[0, 1] = 1) -> None:
        self.stop_bit = validate_integer(stop_bit)
        self.stop_bit = validate_choice(self.stop_bit, (0, 1))

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
        integer = validate_positive_integer(integer)
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
        return validate_positive_integer(integer)

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
