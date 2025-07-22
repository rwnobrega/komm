from functools import partial
from typing import Literal, cast

import numpy as np
import numpy.typing as npt


def bits_to_int(
    input: npt.ArrayLike,
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> int | npt.NDArray[np.integer]:
    r"""
    Converts a bit array to its integer representation.

    Parameters:
        input: The input bit array. Must be an array with elements in the set $\\{ 0, 1 \\}$, with the bit sequences in the last axis.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        output: The integer representation of the input bit array. Has the same shape as the input, but with the last dimension removed.

    Examples:
        >>> komm.bits_to_int([0, 0, 0, 0, 1, 0], bit_order="LSB-first")
        16

        >>> komm.bits_to_int([0, 0, 0, 0, 1, 0], bit_order="MSB-first")
        2

        >>> komm.bits_to_int([[0, 0], [1, 0], [0, 1], [1, 1]])
        array([0, 1, 2, 3])

        >>> komm.bits_to_int([[0, 0], [1, 0], [0, 1], [1, 1]], bit_order="MSB-first")
        array([0, 2, 1, 3])
    """
    if np.ndim(input) == 1:
        if bit_order == "LSB-first":
            input = np.asarray(input, dtype=np.uint8)
        elif bit_order == "MSB-first":
            input = np.flip(input, axis=-1)
        else:
            raise ValueError("'bit_order' must be in {'LSB-first', 'MSB-first'}")
        packed = np.packbits(input, bitorder="little")
        return int.from_bytes(packed, byteorder="little")
    return np.apply_along_axis(partial(bits_to_int, bit_order=bit_order), -1, input)


def int_to_bits(
    input: npt.ArrayLike,
    width: int,
    bit_order: Literal["LSB-first", "MSB-first"] = "LSB-first",
) -> npt.NDArray[np.integer]:
    r"""
    Converts an integer, or array of integers, to their bit representations.

    Parameters:
        input: The input integer, or array of integers.

        width: The width of the bit representation.

        bit_order: Bit order convention. Must be either `"LSB-first"` (least significant bit in the first position) or `"MSB-first"` (most significant bit in the first position). The default value is `"LSB-first"`.

    Returns:
        output: The bit representation of the input, with the bit sequences in the last axis. Has the same shape as the input, but with a new last dimension of size `width` appended.

    Examples:
        >>> komm.int_to_bits(2, width=6, bit_order="LSB-first")
        array([0, 1, 0, 0, 0, 0])

        >>> komm.int_to_bits(2, width=6, bit_order="MSB-first")
        array([0, 0, 0, 0, 1, 0])

        >>> komm.int_to_bits([0, 1, 2, 3], width=4)
        array([[0, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [1, 1, 0, 0]])

        >>> komm.int_to_bits([0, 1, 2, 3], width=4, bit_order="MSB-first")
        array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 1, 1]])
    """
    if np.ndim(input) == 0:
        input = cast(int, input)
        if bit_order == "LSB-first":
            bit_list = [(input >> i) % 2 for i in range(width)]
        elif bit_order == "MSB-first":
            bit_list = [(input >> (width - 1 - i)) % 2 for i in range(width)]
        else:
            raise ValueError("'bit_order' must be in {'LSB-first', 'MSB-first'}")
        return np.array(bit_list, dtype=int)
    input = np.asarray(input, dtype=object)
    bits = np.array(
        [int_to_bits(x, width=width, bit_order=bit_order) for x in input.ravel()],
        dtype=int,
    )
    return bits.reshape(input.shape + (width,))
