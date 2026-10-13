from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import chain, islice
from operator import index
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

from .._lossless_coding.util import Word
from .._util.validators import validate_integer_array
from ..types import Array1D


class IntegerCode(ABC):
    r"""
    Abstract base class for integer codes, which assign to each positive integer a binary codeword whose end can be recognized without knowing its length in advance.
    """

    @abstractmethod
    def encode_single(self, integer: SupportsIndex) -> Word:
        r"""
        Encodes a single integer into its codeword.

        Parameters:
            integer: The integer to be encoded. Must be in the domain of the code.

        Returns:
            codeword: The codeword of the integer. It is a tuple of bits.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_single(self, bits: Iterator[SupportsIndex]) -> int:
        r"""
        Decodes a single codeword from a bit iterator. This method consumes exactly the bits of one codeword, leaving the iterator at the boundary with the next one; any remaining bits are left untouched.

        Parameters:
            bits: An iterator of bits starting with a complete codeword. It must be an iterator (as returned by `iter`), not a general iterable, since advancing it is part of the contract.

        Returns:
            integer: The decoded integer.

        Notes:
            A ValueError is raised if the iterator is exhausted mid-codeword or if an invalid bit is found.
        """
        raise NotImplementedError

    def length(self, integer: SupportsIndex) -> int:
        r"""
        Returns the codeword length $\ell(n)$ for a given integer $n$ in the domain of the code.
        """
        return len(self.encode_single(integer))

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Encodes a sequence of integers.

        Parameters:
            input: The sequence of integers to be encoded. Must be a 1D-array of positive integers.

        Returns:
            output: The concatenation of the codewords. It is a 1D-array of bits.
        """
        input = validate_integer_array(input, low=1, ndim=1)
        output = [bit for n in input.tolist() for bit in self.encode_single(n)]
        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Decodes a sequence of bits.

        Parameters:
            input: The sequence of bits to be decoded. Must be a 1D-array of bits. Also, the sequence must be a concatenation of codewords (i.e., the output of the `encode` method).

        Returns:
            output: The sequence of decoded integers. It is a 1D-array of positive integers.
        """
        input = validate_integer_array(input, low=0, high=2, ndim=1)
        it = iter(input.tolist())
        output = [self.decode_single(chain([first], it)) for first in it]
        return np.array(output, dtype=int)


def take(bits: Iterator[SupportsIndex], num: int) -> list[int]:
    chunk = [index(bit) for bit in islice(bits, num)]
    if len(chunk) < num:
        raise ValueError("input contains an incomplete codeword")
    return chunk
