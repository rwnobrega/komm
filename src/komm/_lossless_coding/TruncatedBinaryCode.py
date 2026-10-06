from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

from .._integer_coding.base import SelfDelimitingCode, take
from .._util.bit_operations import from_binary, to_binary
from .._util.validators import validate_index


@dataclass
class TruncatedBinaryCode(SelfDelimitingCode):
    r"""
    Truncated binary code. It is a code for integers in $[0 : M)$, where $M \geq 2$ is a given cardinality. Let $k = \lfloor \log_2 M \rfloor$ and $u = 2^{k+1} - M$. The codeword for an integer $n \in [0 : M)$ is the $k$-bit binary representation of $n$, if $n < u$, or the $(k + 1)$-bit binary representation of $n + u$, otherwise. For more details, see [Wikipedia: Truncated binary encoding](https://en.wikipedia.org/wiki/Truncated_binary_encoding).

    The table below shows the codewords for small values of $M$.

    | $n$ | $M = 4$ | $M = 5$ | $M = 6$ | $M = 7$ | $M = 8$ |
    | :-: | ------- | ------- | ------- | ------- | ------- |
    | $0$ | `00`    | `00`    | `00`    | `00`    | `000`   |
    | $1$ | `01`    | `01`    | `01`    | `010`   | `001`   |
    | $2$ | `10`    | `10`    | `100`   | `011`   | `010`   |
    | $3$ | `11`    | `110`   | `101`   | `100`   | `011`   |
    | $4$ |         | `111`   | `110`   | `101`   | `100`   |
    | $5$ |         |         | `111`   | `110`   | `101`   |
    | $6$ |         |         |         | `111`   | `110`   |
    | $7$ |         |         |         |         | `111`   |

    Notes:
        - For $M$ a power of $2$ it reduces to the fixed-length binary code.
        - It coincides with the [Huffman code](/ref/HuffmanCode) for the uniform pmf, with canonical assignment.

    Parameters:
        cardinality: The cardinality $M$ of the code. Must satisfy $M \geq 2$.
    """

    cardinality: int

    def __post_init__(self) -> None:
        self.cardinality = validate_index(self.cardinality, low=2)
        self._k = self.cardinality.bit_length() - 1
        self._u = 2 ** (self._k + 1) - self.cardinality

    def _validate(self, integer: SupportsIndex) -> int:
        integer = index(integer)
        if not 0 <= integer < self.cardinality:
            raise ValueError("input contains an out-of-range entry")
        return integer

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> code.encode_single(2)
            [1, 0]
            >>> code.encode_single(3)
            [1, 1, 0]
        """
        integer = self._validate(integer)
        u, k = self._u, self._k
        if integer < u:
            return to_binary(integer, width=k, bit_order="MSB-first")
        return to_binary(integer + u, width=k + 1, bit_order="MSB-first")

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
        integer = from_binary(take(bits, k), bit_order="MSB-first")
        if integer < u:
            return integer
        return 2 * integer + from_binary(take(bits, 1), bit_order="MSB-first") - u

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.TruncatedBinaryCode(5)
            >>> code.length(2), code.length(3)
            (2, 3)
        """
        u, k = self._u, self._k
        integer = self._validate(integer)
        return k if integer < u else k + 1

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
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
