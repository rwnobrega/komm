from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import SupportsIndex

from .. import abc
from .._lossless_coding.TruncatedBinaryCode import TruncatedBinaryCode
from .._util.validators import validate_positive_integer


@dataclass
class EscapeCode(abc.IntegerCode):
    r"""
    Escape code. It is an integer code with domain the positive integers. Let $M \geq 1$ be the *divisor*. For an integer $n$, let $q$ and $r$ be the quotient and the remainder of the division of $n - 1$ by $M$. The codeword for $n$ is a sequence of blocks, each one a [truncated binary codeword](/ref/TruncatedBinaryCode) with cardinality $M + 1$: $q$ blocks of $M$, called the *escape*, followed by the block of $r$.

    The table below shows the codewords for the first integers, with a space between the blocks.

    | $n$ | $M = 1$           | $M = 2$       | $M = 3$    | $M = 4$   |
    | :-: | ----------------- | ------------- | ---------- | --------- |
    | $1$ | `0`               | `0`           | `00`       | `00`      |
    | $2$ | `1 0`             | `10`          | `01`       | `01`      |
    | $3$ | `1 1 0`           | `11 0`        | `10`       | `10`      |
    | $4$ | `1 1 1 0`         | `11 10`       | `11 00`    | `110`     |
    | $5$ | `1 1 1 1 0`       | `11 11 0`     | `11 01`    | `111 00`  |
    | $6$ | `1 1 1 1 1 0`     | `11 11 10`    | `11 10`    | `111 01`  |
    | $7$ | `1 1 1 1 1 1 0`   | `11 11 11 0`  | `11 11 00` | `111 10`  |
    | $8$ | `1 1 1 1 1 1 1 0` | `11 11 11 10` | `11 11 01` | `111 110` |

    Notes:
        - For $M = 1$ it reduces to the [unary code](/ref/UnaryCode) with stop bit $0$.
        - For $M = 2^b - 1$ all blocks have $b$ bits.
        - The [Golomb code](/ref/GolombCode) uses the same division, but writes $q$ in unary.
        - The [taboo code](/ref/TabooCode) uses the same blocks, but reserves the block of $0$ to end the codeword, instead of the block of $M$ to extend it.

    Parameters:
        divisor: The divisor $M$. Must satisfy $M \geq 1$.
    """

    divisor: int

    def __post_init__(self) -> None:
        self.divisor = validate_positive_integer(self.divisor, "divisor")
        self._block_code = TruncatedBinaryCode(self.divisor + 1)

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.EscapeCode(3)
            >>> code.encode_single(4)
            [1, 1, 0, 0]
        """
        integer = validate_positive_integer(integer, "integer")
        q, r = divmod(integer - 1, self.divisor)
        bits: list[int] = []
        for value in [self.divisor] * q + [r]:
            bits += self._block_code.encode_single(value)
        return bits

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.EscapeCode(3)
            >>> bits = iter([1, 1, 0, 0, 0, 0])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [0, 0]
        """
        integer = 0
        while True:
            value = self._block_code.decode_single(bits)
            if value < self.divisor:
                return integer + value + 1
            integer += value

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.EscapeCode(3)
            >>> code.length(4)
            4
        """
        integer = validate_positive_integer(integer, "integer")
        q, r = divmod(integer - 1, self.divisor)
        escape_length = self._block_code.length(self.divisor)
        return q * escape_length + self._block_code.length(r)

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EscapeCode(3)
            >>> list(code.encode([4, 1, 3]))
            [1, 1, 0, 0, 0, 0, 1, 0]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EscapeCode(3)
            >>> list(code.decode([1, 1, 0, 0, 0, 0, 1, 0]))
            [4, 1, 3]
        """
        return super().decode(input)
