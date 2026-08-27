from collections.abc import Iterable, Iterator

from .. import abc
from .base import validate_positive


class UnaryCode(abc.IntegerCode):
    r"""
    Unary code. It is an integer code with domain the positive integers. The codeword for an integer $n$ consists of $n - 1$ zeros followed by a single $1$. For more details, see <cite>MacK03, Ch. 7</cite>.
    """

    def encode_single(self, integer: int) -> list[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.encode_single(4)
            [0, 0, 0, 1]
        """
        validate_positive(integer)
        return [0] * (integer - 1) + [1]

    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> bits = iter([0, 0, 0, 1, 1, 0])
            >>> code.decode_single(bits)
            4
            >>> list(bits)  # Iterator is left at codeword boundary
            [1, 0]
        """
        for pos, bit in enumerate(bits):
            if bit == 1:
                return pos + 1
            if bit != 0:
                raise ValueError(f"invalid bit in input: {bit}")
        raise ValueError("input contains an incomplete codeword")

    def length(self, integer: int) -> int:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.length(4)
            4
        """
        validate_positive(integer)
        return integer

    def encode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> list(code.encode([4, 1, 3]))
            [0, 0, 0, 1, 1, 0, 0, 1]
        """
        return super().encode(input)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> list(code.decode([0, 0, 0, 1, 1, 0, 0, 1]))
            [4, 1, 3]
        """
        return super().decode(input)
