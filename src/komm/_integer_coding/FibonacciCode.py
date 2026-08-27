from collections.abc import Iterable, Iterator
from functools import cache

from .. import abc
from .base import validate_positive


class FibonacciCode(abc.IntegerCode):
    r"""
    Fibonacci code. It is an integer code with domain the positive integers. For the definition of this code, see [Wikipedia: Fibonacci coding](https://en.wikipedia.org/wiki/Fibonacci_coding).
    """

    def encode_single(self, integer: int) -> list[int]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.encode_single(4)
            [1, 0, 1, 1]
        """
        validate_positive(integer)
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

    def length(self, integer: int) -> int:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.length(4)
            4
        """
        validate_positive(integer)
        return super().length(integer)

    def encode(self, input: Iterable[int]) -> Iterator[int]:
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
