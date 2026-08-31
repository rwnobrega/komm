from collections.abc import Iterable, Iterator

from .. import abc
from .._util.bit_operations import from_binary, to_binary
from .base import take, validate_positive
from .UnaryCode import UnaryCode


class EliasGammaCode(abc.IntegerCode):
    r"""
    Elias gamma code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of the [unary codeword](/ref/UnaryCode) for the number of bits of $n$ followed by the binary representation of $n$ without its leading one. For more details, see [Wikipedia: Elias gamma coding](https://en.wikipedia.org/wiki/Elias_gamma_coding) or <cite>MacK03, Ch. 7</cite> (therein called code $C_\alpha$).
    """

    unary_code = UnaryCode()

    def encode_single(self, integer: int) -> list[int]:
        r"""
        Examples:
            >>> code = komm.EliasGammaCode()
            >>> code.encode_single(4)
            [0, 0, 1, 0, 0]
        """
        validate_positive(integer)
        binary = to_binary(integer, bit_order="MSB-first")
        return self.unary_code.encode_single(len(binary)) + binary[1:]

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.EliasGammaCode()
            >>> bits = iter([0, 0, 1, 0, 0, 1, 1])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 1]
        """
        length = self.unary_code.decode_single(bits)
        return from_binary([1] + take(bits, length - 1), bit_order="MSB-first")

    def length(self, integer: int) -> int:
        r"""
        Examples:
            >>> code = komm.EliasGammaCode()
            >>> code.length(4)
            5
        """
        validate_positive(integer)
        return 2 * integer.bit_length() - 1

    def encode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasGammaCode()
            >>> list(code.encode([4, 1, 3]))
            [0, 0, 1, 0, 0, 1, 0, 1, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.EliasGammaCode()
            >>> list(code.decode([0, 0, 1, 0, 0, 1, 0, 1, 1]))
            [4, 1, 3]
        """
        return super().decode(input)
