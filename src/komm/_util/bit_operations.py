from collections.abc import Iterable
from operator import index
from typing import Literal

import numpy as np
import numpy.typing as npt

from .validators import (
    validate_binary_array,
    validate_bit_order,
    validate_integer,
    validate_integer_array,
)


def bits_to_int(
    input: npt.ArrayLike,
    width: int,
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> npt.NDArray[np.integer]:
    r"""
    Converts a bit array to its integer representation. The last dimension of the input is split into blocks of `width` bits, each of which is converted to an integer.

    For integers wider than $63$ bits, see [`from_binary`](/ref/from_binary).

    Parameters:
        input: The input bit array. Must be an array with elements in $\\{ 0, 1 \\}$ whose last dimension is a multiple of `width`.

        width: The number of bits per integer. Must be in $[1 : 64)$.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        output: The integer representation of the input bit array. Has the same shape as the input, but with the last dimension contracted by a factor of `width`.

    Examples:
        >>> komm.bits_to_int([0, 0, 0, 0, 1, 0], width=6)
        array([16])

        >>> komm.bits_to_int([0, 0, 0, 0, 1, 0], width=6, bit_order="MSB-first")
        array([2])

        >>> komm.bits_to_int([0, 0, 0, 0, 1, 0], width=3)
        array([0, 2])

        >>> komm.bits_to_int([[0, 0, 1, 1], [0, 1, 0, 1]], width=2)
        array([[0, 3],
               [2, 2]])
    """
    bit_order = validate_bit_order(bit_order)
    width = validate_integer(width, low=1, high=64)
    input = validate_binary_array(input)
    if input.shape[-1] % width != 0:
        raise ValueError(
            f"last dimension of 'input' must be a multiple of {width}"
            f" (got {input.shape[-1]})"
        )
    weights = 1 << np.arange(width)
    if bit_order == "MSB-first":
        weights = weights[::-1]
    blocks = input.reshape(*input.shape[:-1], -1, width)
    return blocks @ weights


def int_to_bits(
    input: npt.ArrayLike,
    width: int,
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> npt.NDArray[np.integer]:
    r"""
    Converts an integer array to its bit representation. Each integer is converted to `width` bits, which are concatenated along the last dimension. This is the inverse of [`bits_to_int`](/ref/bits_to_int).

    For integers wider than $63$ bits, see [`to_binary`](/ref/to_binary).

    Parameters:
        input: The input integer array. Must be an array of integers in $[0 : 2^{\mathtt{width}})$.

        width: The number of bits per integer. Must be in $[0 : 64)$.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        output: The bit representation of the input integer array. Has the same shape as the input, but with the last dimension expanded by a factor of `width`.

    Examples:
        >>> komm.int_to_bits([16], width=6)
        array([0, 0, 0, 0, 1, 0])

        >>> komm.int_to_bits([2], width=6, bit_order="MSB-first")
        array([0, 0, 0, 0, 1, 0])

        >>> komm.int_to_bits([0, 2], width=3)
        array([0, 0, 0, 0, 1, 0])

        >>> komm.int_to_bits([[0, 3], [2, 2]], width=2)
        array([[0, 0, 1, 1],
               [0, 1, 0, 1]])
    """
    bit_order = validate_bit_order(bit_order)
    width = validate_integer(width, low=0, high=64)
    input = validate_integer_array(input, low=0, high=1 << width)
    shifts = np.arange(width)
    if bit_order == "MSB-first":
        shifts = shifts[::-1]
    bits = (input[..., np.newaxis] >> shifts) & 1
    return bits.reshape(*input.shape[:-1], -1)


def to_binary(
    integer: int,
    width: int | None = None,
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> list[int]:
    r"""
    Converts a single integer to its bit representation. Unlike [`int_to_bits`](/ref/int_to_bits), it works with Python integers of arbitrary size.

    Parameters:
        integer: The input integer. Must be non-negative.

        width: The number of bits of the representation. The default value is the number of bits of the input integer.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        bits: The bit representation of the input integer, as a list of bits.

    Examples:
        >>> komm.to_binary(16, width=6)
        [0, 0, 0, 0, 1, 0]

        >>> komm.to_binary(16)
        [0, 0, 0, 0, 1]

        >>> komm.to_binary(16, bit_order="MSB-first")
        [1, 0, 0, 0, 0]

        >>> komm.from_binary(komm.to_binary(2**100))
        1267650600228229401496703205376
    """
    bit_order = validate_bit_order(bit_order)
    integer = validate_integer(integer)
    if width is None:
        width = integer.bit_length()
    integer = validate_integer(
        integer, low=0, high=1 << width, rule="0 <= integer < 2**width"
    )
    bits = [(integer >> i) & 1 for i in range(width)]
    if bit_order == "MSB-first":
        bits.reverse()
    return bits


def from_binary(
    bits: Iterable[int],
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> int:
    r"""
    Converts a bit sequence to a single integer. Unlike [`bits_to_int`](/ref/bits_to_int), it works with Python integers of arbitrary size.

    Parameters:
        bits: The input bit sequence. Must be an iterable with elements in $\\{ 0, 1 \\}$.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        integer: The integer represented by the input bit sequence.

    Examples:
        >>> komm.from_binary([0, 0, 0, 0, 1, 0])
        16

        >>> komm.from_binary([0, 0, 0, 0, 1, 0], bit_order="MSB-first")
        2

        >>> komm.from_binary([0] * 100 + [1])
        1267650600228229401496703205376
    """
    bit_order = validate_bit_order(bit_order)
    bit_list = [index(bit) for bit in bits]
    if bit_order == "LSB-first":
        bit_list.reverse()
    integer = 0
    for bit in bit_list:
        if bit not in {0, 1}:
            raise ValueError(f"invalid bit in input: {bit}")
        integer = 2 * integer + bit
    return integer
