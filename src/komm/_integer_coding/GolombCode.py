from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import SupportsIndex

from .. import abc
from .._lossless_coding.TruncatedBinaryCode import TruncatedBinaryCode
from .._util.validators import validate_integer
from .UnaryCode import UnaryCode


@dataclass
class GolombCode(abc.IntegerCode):
    r"""
    Golomb code. It is an integer code with domain the positive integers. Let $M \geq 1$ be the *divisor*. For an integer $n$, let $q$ and $r$ be the quotient and the remainder of the division of $n - 1$ by $M$. The codeword for $n$ consists of the [unary codeword](/ref/UnaryCode) of $q + 1$ with stop bit $0$ (that is, $q$ ones followed by a single zero), followed by the [truncated binary codeword](/ref/TruncatedBinaryCode) of $r$ with cardinality $M$ (empty if $M = 1$). For more details, see <cite>Gol66</cite> or <cite>Say06, Sec. 3.5</cite>, where the domain is the non-negative integers, so that the codeword for $n$ here is their codeword for $n - 1$.

    The table below shows the codewords for the first integers, with a space between the two parts.

    | $n$ | $M = 1$    | $M = 2$  | $M = 3$  | $M = 4$ |
    | :-: | ---------- | -------- | -------- | ------- |
    | $1$ | `0`        | `0 0`    | `0 0`    | `0 00`  |
    | $2$ | `10`       | `0 1`    | `0 10`   | `0 01`  |
    | $3$ | `110`      | `10 0`   | `0 11`   | `0 10`  |
    | $4$ | `1110`     | `10 1`   | `10 0`   | `0 11`  |
    | $5$ | `11110`    | `110 0`  | `10 10`  | `10 00` |
    | $6$ | `111110`   | `110 1`  | `10 11`  | `10 01` |
    | $7$ | `1111110`  | `1110 0` | `110 0`  | `10 10` |
    | $8$ | `11111110` | `1110 1` | `110 10` | `10 11` |

    Notes:
        - For $M = 1$ it reduces to the [unary code](/ref/UnaryCode) with stop bit $0$.
        - For $M$ a power of $2$ it is also known as the *Rice code*.

    Parameters:
        divisor: The divisor $M$. Must satisfy $M \geq 1$.
    """

    divisor: int

    def __post_init__(self) -> None:
        self.divisor = validate_integer(self.divisor, "divisor", low=1)
        self._unary_code = UnaryCode(stop_bit=0)
        self._remainder_code = (
            TruncatedBinaryCode(self.divisor) if self.divisor >= 2 else None
        )

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.GolombCode(3)
            >>> code.encode_single(4)
            [1, 0, 0]
        """
        integer = validate_integer(integer, "integer", low=1)
        q, r = divmod(integer - 1, self.divisor)
        bits = self._unary_code.encode_single(q + 1)
        if self._remainder_code is not None:
            bits += self._remainder_code.encode_single(r)
        return bits

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.GolombCode(3)
            >>> bits = iter([1, 0, 0, 0, 0])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [0, 0]
        """
        q = self._unary_code.decode_single(bits) - 1
        r = 0
        if self._remainder_code is not None:
            r = self._remainder_code.decode_single(bits)
        return q * self.divisor + r + 1

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.GolombCode(3)
            >>> code.length(4)
            3
        """
        integer = validate_integer(integer, "integer", low=1)
        q, r = divmod(integer - 1, self.divisor)
        if self._remainder_code is None:
            return q + 1
        return q + 1 + self._remainder_code.length(r)

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.GolombCode(3)
            >>> list(code.encode([4, 1, 3]))
            [1, 0, 0, 0, 0, 0, 1, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.GolombCode(3)
            >>> list(code.decode([1, 0, 0, 0, 0, 0, 1, 1]))
            [4, 1, 3]
        """
        return super().decode(input)
