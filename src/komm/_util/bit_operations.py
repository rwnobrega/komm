from typing import cast

import numpy as np
import numpy.typing as npt


def bits_to_int(input: npt.ArrayLike) -> int | npt.NDArray[np.integer]:
    r"""
    Converts a bit array to its integer representation (LSB first).

    Parameters:
        input: An $N$-dimensional array of $0$s and $1$s. The least significant bit (LSB) is the first element in the last dimension.

    Returns:
        An integer or an $(N-1)$-dimensional array of integers. The last dimension of the input is collapsed into an integer representation while all preceding dimensions are preserved.

    Examples:
        >>> komm.bits_to_int([0, 0, 0, 0, 1])
        16

        >>> komm.bits_to_int([0, 1, 0, 1, 1])
        26

        >>> komm.bits_to_int([0, 1, 0, 1, 1, 0, 0, 0])
        26

        >>> komm.bits_to_int([[0, 0], [1, 0], [0, 1], [1, 1]])  # Each row is independently converted to an integer
        array([0, 1, 2, 3])
    """
    if np.ndim(input) > 1:
        return np.apply_along_axis(bits_to_int, -1, input)
    input = np.asarray(input, dtype=np.uint8)
    packed = np.packbits(input, bitorder="little")
    return int.from_bytes(packed, byteorder="little")


def int_to_bits(input: npt.ArrayLike, width: int) -> npt.NDArray[np.integer]:
    r"""
    Converts an integer, or array of integers, to their bit representations (LSB first).

    Parameters:
        input: An integer or an $N$-dimensional array of integers.

        width: The width of the bit representation.

    Returns:
        An $(N+1)$-dimensional array of $0$s and $1$s, where the last dimension contains the bit representation of the input, with the least significant bit (LSB) as the first element.

    Examples:
        >>> komm.int_to_bits(16, width=5)
        array([0, 0, 0, 0, 1])

        >>> komm.int_to_bits(26, width=5)
        array([0, 1, 0, 1, 1])

        >>> komm.int_to_bits(26, width=8)
        array([0, 1, 0, 1, 1, 0, 0, 0])

        >>> komm.int_to_bits([0, 1, 2, 3], width=2)
        array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]])

        >>> komm.int_to_bits([0, 1, 2, 3], width=4)
        array([[0, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [1, 1, 0, 0]])

        >>> komm.int_to_bits([[0, 1], [2, 3]], width=2)
        array([[[0, 0],
                [1, 0]],
        <BLANKLINE>
               [[0, 1],
                [1, 1]]])
    """
    if np.ndim(input) > 0:
        input = np.asarray(input, dtype=object)
        bits = np.array([int_to_bits(x, width) for x in input.flat])
        return bits.reshape(input.shape + (width,))
    input = cast(int, input)
    return np.array([(input >> i) & 1 for i in range(width)])
