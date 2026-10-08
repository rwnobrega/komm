from collections.abc import Iterable, Iterator
from typing import SupportsIndex

from .. import abc
from .._util.bit_operations import from_binary, to_binary
from .._util.validators import validate_positive_integer
from .base import take
from .EliasGammaCode import EliasGammaCode


class EliasDeltaCode(abc.IntegerCode):
    r"""
    Elias delta code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of the [Elias gamma codeword](/ref/EliasGammaCode) for the number of bits of $n$ followed by the binary representation of $n$ without its leading one. For more details, see [Wikipedia: Elias delta coding](https://en.wikipedia.org/wiki/Elias_delta_coding) or <cite>MacK03, Ch. 7</cite> (therein called code $C_\beta$).

    The table below shows the codewords for the first integers, with a space between the two parts.

    | $n$ | Codeword    |
    | :-: | ----------- |
    | $1$ | `1`         |
    | $2$ | `010 0`     |
    | $3$ | `010 1`     |
    | $4$ | `011 00`    |
    | $5$ | `011 01`    |
    | $6$ | `011 10`    |
    | $7$ | `011 11`    |
    | $8$ | `00100 000` |
    """

    gamma_code = EliasGammaCode()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def encode_single(self, integer: SupportsIndex) -> list[int]:
        r"""
        Examples:
            >>> code = komm.EliasDeltaCode()
            >>> code.encode_single(4)
            [0, 1, 1, 0, 0]
        """
        integer = validate_positive_integer(integer, "integer")
        binary = to_binary(integer, bit_order="MSB-first")
        return self.gamma_code.encode_single(len(binary)) + binary[1:]

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.EliasDeltaCode()
            >>> bits = iter([0, 1, 1, 0, 0, 1, 1])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 1]
        """
        length = self.gamma_code.decode_single(bits)
        return from_binary([1] + take(bits, length - 1), bit_order="MSB-first")

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Examples:
            >>> code = komm.EliasDeltaCode()
            >>> code.length(4)
            5
        """
        integer = validate_positive_integer(integer, "integer")
        num_bits = integer.bit_length()
        return self.gamma_code.length(num_bits) + num_bits - 1

    def encode(self, input: Iterable[SupportsIndex]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasDeltaCode()
            >>> list(code.encode([4, 1, 3]))
            [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasDeltaCode()
            >>> list(code.decode([0, 1, 1, 0, 0, 1, 0, 1, 0, 1]))
            [4, 1, 3]
        """
        return super().decode(input)
