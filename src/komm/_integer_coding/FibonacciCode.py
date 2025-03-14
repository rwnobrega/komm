from functools import cache

import numpy as np
import numpy.typing as npt

from . import base


class FibonacciCode(base.IntegerCode):
    r"""
    Fibonacci code. It is an integer code. For the definition of this code, see [Wikipedia: Fibonacci coding](https://en.wikipedia.org/wiki/Fibonacci_coding).
    """

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.encode([4, 1, 3])
            array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        """
        input = np.asarray(input)
        if input.size == 0:
            return np.array([], dtype=int)
        return np.concatenate([fibonacci_encode(i) for i in input])

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.FibonacciCode()
            >>> code.decode([1, 0, 1, 1, 1, 1, 0, 0, 1, 1])
            array([4, 1, 3])
        """
        input = np.asarray(input)
        output: list[int] = []
        i = 0
        while i < input.size:
            j = i
            while not input[j] == input[j + 1] == 1:
                j += 1
            output.append(fibonacci_decode(list(input[i : j + 2])))
            i = j + 2
        return np.array(output)


@cache
def fibonacci(n: int) -> int:
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_encode(integer: int) -> list[int]:
    positions: list[int] = []
    while integer > 0:
        i = 0
        while fibonacci(i + 1) <= integer:
            i += 1
        integer -= fibonacci(i)
        positions.append(i - 2)
    last = max(positions) + 1
    return [1 if i in positions else 0 for i in range(last)] + [1]


def fibonacci_decode(bits: list[int]) -> int:
    return sum(fibonacci(i + 2) for i, bit in enumerate(bits[:-1]) if bit)
