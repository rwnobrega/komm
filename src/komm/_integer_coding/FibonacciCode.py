from collections.abc import Iterable, Iterator
from functools import cache
from typing import SupportsIndex

from .. import abc
from .._util.validators import validate_positive_integer


class FibonacciCode(abc.IntegerCode):
    r"""
    Fibonacci code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of the [Zeckendorf representation](https://en.wikipedia.org/wiki/Zeckendorf%27s_theorem) of $n$ (a sum of non-consecutive Fibonacci numbers $1, 2, 3, 5, 8, \ldots$), least significant bit first, followed by an extra $1$. For more details, see [Wikipedia: Fibonacci coding](https://en.wikipedia.org/wiki/Fibonacci_coding).

    The table below shows the codewords for the first integers, with a space between the two parts.

    | $n$ | Codeword  |
    | :-: | --------- |
    | $1$ | `1 1`     |
    | $2$ | `01 1`    |
    | $3$ | `001 1`   |
    | $4$ | `101 1`   |
    | $5$ | `0001 1`  |
    | $6$ | `1001 1`  |
    | $7$ | `0101 1`  |
    | $8$ | `00001 1` |
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.encode_single(4)
            [1, 0, 1, 1]
        """
        integer = validate_positive_integer(integer)
        top = 2
        while fibonacci(top + 1) <= integer:
            top += 1
        bits = [0] * (top - 1)
        for i in range(top, 1, -1):
            if fibonacci(i) <= integer:
                bits[i - 2] = 1
                integer -= fibonacci(i)
        return bits + [1]

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> bits = iter([1, 0, 1, 1, 0, 1])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [0, 1]
        """
        integer = 0
        last = 0
        for pos, bit in enumerate(bits):
            if bit == 1:
                if last == 1:
                    return integer
                integer += fibonacci(pos + 2)
            elif bit != 0:
                raise ValueError(f"invalid bit in input: {bit}")
            last = bit
        raise ValueError("input contains an incomplete codeword")

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.length(4)
            4
        """
        return super().length(integer)

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> list(code.encode([4, 1, 3]))
            [1, 0, 1, 1, 1, 1, 0, 0, 1, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> list(code.decode([1, 0, 1, 1, 1, 1, 0, 0, 1, 1]))
            [4, 1, 3]
        """
        return super().decode(input)


@cache
def fibonacci(n: int) -> int:
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
