from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import chain, islice
from operator import index


class IntegerCode(ABC):
    @abstractmethod
    def encode_single(self, integer: int) -> list[int]:
        r"""
        Encodes a single integer into its codeword.

        Parameters:
            integer: The integer to be encoded. Must be positive.

        Returns:
            bits: The codeword of the integer, as a list of bits.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_single(self, bits: Iterator[int]) -> int:
        r"""
        Decodes a single codeword from a bit iterator. This method consumes exactly the bits of one codeword, leaving the iterator at the boundary with the next one; any remaining bits are left untouched.

        Parameters:
            bits: An iterator of bits starting with a complete codeword. It must be an iterator (as returned by `iter`), not a general iterable, since advancing it is part of the contract.

        Returns:
            integer: The decoded integer.

        Notes:
            A ValueError is raised if the iterator is exhausted mid-codeword or if an invalid bit is found. In contrast, `next(self.decode(bits))` behaves identically except on an exhausted iterator, where it raises StopIteration (end of data) instead of ValueError (malformed data).
        """
        raise NotImplementedError

    def length(self, integer: int) -> int:
        r"""
        Returns the codeword length $\ell(n)$ for a given positive integer $n$.
        """
        return len(self.encode_single(integer))

    def encode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Lazily encodes an iterable of positive integers.

        Parameters:
            input: The integers to be encoded. Must all be positive.

        Returns:
            output: An iterator over the bits of the concatenated codewords.
        """
        for integer in input:
            yield from self.encode_single(integer)

    def decode(self, input: Iterable[int]) -> Iterator[int]:
        r"""
        Lazily decodes an iterable of bits.

        Note:
            Decoding is lazy: invalid or truncated input only raises ValueError when the offending codeword is consumed.

        Parameters:
            input: The bits to be decoded. Must be a concatenation of codewords, possibly partial.

        Returns:
            output: An iterator over the decoded positive integers.
        """
        it = iter(input)
        for first in it:
            yield self.decode_single(chain([first], it))


def validate_positive(integer: int) -> None:
    if not integer > 0:
        raise ValueError("input contains a non-positive entry")


def take(bits: Iterator[int], num: int) -> list[int]:
    chunk = list(islice(bits, num))
    if len(chunk) < num:
        raise ValueError("input contains an incomplete codeword")
    return chunk


def to_binary(integer: int) -> list[int]:
    integer = index(integer)  # Aceita int-like (e.g. np.int64), rejeita float
    return [(integer >> i) & 1 for i in range(integer.bit_length() - 1, -1, -1)]


def from_binary(bits: Iterable[int]) -> int:
    integer = 0
    for bit in bits:
        if bit != 0 and bit != 1:
            raise ValueError(f"invalid bit in input: {bit}")
        integer = 2 * integer + bit
    return integer
