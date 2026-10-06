from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

from .. import abc
from .._lossless_coding.TruncatedBinaryCode import TruncatedBinaryCode
from .base import validate_positive


@dataclass
class TabooCode(abc.IntegerCode):
    r"""
    Taboo code. It is an integer code with domain the positive integers. Let $M \geq 1$ be the *base*. The codeword for an integer $n$ is a sequence of blocks, each one a [truncated binary codeword](/ref/TruncatedBinaryCode) with cardinality $M + 1$: the blocks of the digits of $n - 1$ in [bijective base $M$](https://en.wikipedia.org/wiki/Bijective_numeration), which range from $1$ to $M$, most significant first, followed by the block of $0$, called the *taboo*.

    This is a variation of the block taboo codes of <cite>SM10, Sec. 3.16</cite>, which have blocks of $b$ bits (that is, $M = 2^b - 1$) and do not use the taboo alone as a codeword. Here, the taboo alone is the codeword for $n = 1$, which makes the code complete; the codewords for $n \geq 2$ are those of <cite>SM10</cite>, in the same order. The codes $C_3$, $C_7$, and $C_{15}$ of <cite>MacK03, Ch. 7</cite> are similar, but write $n$ in the usual base $M$, without leading zeros, and are therefore not complete.

    The table below shows the codewords for the first integers, with a space between the blocks.

    | $n$ | $M = 1$           | $M = 2$      | $M = 3$    | $M = 4$     |
    | :-: | ----------------- | ------------ | ---------- | ----------- |
    | $1$ | `0`               | `0`          | `00`       | `00`        |
    | $2$ | `1 0`             | `10 0`       | `01 00`    | `01 00`     |
    | $3$ | `1 1 0`           | `11 0`       | `10 00`    | `10 00`     |
    | $4$ | `1 1 1 0`         | `10 10 0`    | `11 00`    | `110 00`    |
    | $5$ | `1 1 1 1 0`       | `10 11 0`    | `01 01 00` | `111 00`    |
    | $6$ | `1 1 1 1 1 0`     | `11 10 0`    | `01 10 00` | `01 01 00`  |
    | $7$ | `1 1 1 1 1 1 0`   | `11 11 0`    | `01 11 00` | `01 10 00`  |
    | $8$ | `1 1 1 1 1 1 1 0` | `10 10 10 0` | `10 01 00` | `01 110 00` |

    Notes:
        - For $M = 1$ it reduces to the [unary code](/ref/UnaryCode) with stop bit $0$.
        - For $M = 2$ it has the same codeword lengths as the [Elias gamma code](/ref/EliasGammaCode).
        - The [escape code](/ref/EscapeCode) uses the same blocks, but reserves the block of $M$ to extend the codeword, instead of the block of $0$ to end it.

    Parameters:
        base: The base $M$. Must satisfy $M \geq 1$.
    """

    base: int

    def __post_init__(self) -> None:
        self.base = index(self.base)
        if not self.base >= 1:
            raise ValueError("'base' must be at least 1")
        self._block_code = TruncatedBinaryCode(self.base + 1)

    def _digits(self, integer: int) -> list[int]:
        # Bijective numeration, most significant first
        digits: list[int] = []
        integer -= 1
        while integer > 0:
            integer, digit = divmod(integer - 1, self.base)
            digits.append(digit + 1)
        return digits[::-1]

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.TabooCode(3)
            >>> code.encode_single(4)
            [1, 1, 0, 0]
        """
        integer = validate_positive(integer)
        bits: list[int] = []
        for digit in self._digits(integer) + [0]:
            bits += self._block_code.encode_single(digit)
        return bits

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.TabooCode(3)
            >>> bits = iter([1, 1, 0, 0, 0, 0])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [0, 0]
        """
        integer = 0
        while True:
            digit = self._block_code.decode_single(bits)
            if digit == 0:
                return integer + 1
            integer = integer * self.base + digit

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.TabooCode(3)
            >>> code.length(4)
            4
        """
        integer = validate_positive(integer)
        digits = self._digits(integer) + [0]
        return sum(self._block_code.length(digit) for digit in digits)

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.TabooCode(3)
            >>> list(code.encode([4, 1, 3]))
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.TabooCode(3)
            >>> list(code.decode([1, 1, 0, 0, 0, 0, 1, 0, 0, 0]))
            [4, 1, 3]
        """
        return super().decode(input)
