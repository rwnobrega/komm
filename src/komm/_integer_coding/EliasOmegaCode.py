from collections.abc import Iterable, Iterator
from typing import SupportsIndex

from .. import abc
from .._util.bit_operations import from_binary, to_binary
from .._util.validators import validate_positive_integer
from .base import take


class EliasOmegaCode(abc.IntegerCode):
    r"""
    Elias omega code. It is an integer code with domain the positive integers. The codeword for an integer $n$ is built from right to left: it ends with a single $0$ and, while $n > 1$, the binary representation of $n$ is prepended, and $n$ is replaced by its number of bits minus one. For more details, see [Wikipedia: Elias omega coding](https://en.wikipedia.org/wiki/Elias_omega_coding) or <cite>MacK03, Ch. 7</cite> (therein called code $C_\omega$).

    The table below shows the codewords for the first integers, with a space between the parts.

    | $n$ | Codeword    |
    | :-: | ----------- |
    | $1$ | `0`         |
    | $2$ | `10 0`      |
    | $3$ | `11 0`      |
    | $4$ | `10 100 0`  |
    | $5$ | `10 101 0`  |
    | $6$ | `10 110 0`  |
    | $7$ | `10 111 0`  |
    | $8$ | `11 1000 0` |
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.EliasOmegaCode()
            >>> code.encode_single(4)
            [1, 0, 1, 0, 0, 0]
        """
        integer = validate_positive_integer(integer)
        bits = [0]
        while integer > 1:
            binary = to_binary(integer, bit_order="MSB-first")
            bits = binary + bits
            integer = len(binary) - 1
        return bits

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.EliasOmegaCode()
            >>> bits = iter([1, 0, 1, 0, 0, 0, 1, 1])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 1]
        """
        integer = 1
        while True:
            (bit,) = take(bits, 1)
            if bit == 0:
                return integer
            if bit != 1:
                raise ValueError(f"invalid bit in input: {bit}")
            integer = from_binary([1] + take(bits, integer), bit_order="MSB-first")

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.EliasOmegaCode()
            >>> code.length(4)
            6
        """
        integer = validate_positive_integer(integer)
        length = 1
        while integer > 1:
            num_bits = integer.bit_length()
            length += num_bits
            integer = num_bits - 1
        return length

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasOmegaCode()
            >>> list(code.encode([4, 1, 3]))
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 0]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasOmegaCode()
            >>> list(code.decode([1, 0, 1, 0, 0, 0, 0, 1, 1, 0]))
            [4, 1, 3]
        """
        return super().decode(input)
