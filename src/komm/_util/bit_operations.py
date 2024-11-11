from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt


def binlist2int(binlist: Iterable[int]) -> int:
    r"""
    Converts a bit array to its integer representation (LSB first).

    Parameters:
        binlist: A list or array of $0$'s and $1$'s whose $i$-th element stands for the coefficient of $2^i$ in the binary representation of the output integer.

    Returns:
        integer: The integer representation of the input bit array.

    Examples:
        >>> komm.binlist2int([0, 0, 0, 0, 1])
        16

        >>> komm.binlist2int([0, 1, 0, 1, 1])
        26

        >>> komm.binlist2int([0, 1, 0, 1, 1, 0, 0, 0])
        26
    """
    return sum(1 << i for (i, b) in enumerate(binlist) if b != 0)


def int2binlist(integer: int, width: Optional[int] = None) -> npt.NDArray[np.int_]:
    r"""
    Converts an integer to its bit array representation (LSB first).

    Parameters:
        integer: The input integer. May be any nonnegative integer.
        width: If this parameter is specified, the output will be filled with zeros on the right so that its length will be the specified value.

    Returns:
        binlist (Array1D[int]): An array of $0$'s and $1$'s whose $i$-th element stands for the coefficient of $2^i$ in the binary representation of the input integer.

    Examples:
        >>> komm.int2binlist(16)
        array([0, 0, 0, 0, 1])

        >>> komm.int2binlist(26)
        array([0, 1, 0, 1, 1])

        >>> komm.int2binlist(26, width=8)
        array([0, 1, 0, 1, 1, 0, 0, 0])
    """
    if width is None:
        width = max(integer.bit_length(), 1)
    return np.array([(integer >> i) & 1 for i in range(width)], dtype=int)


def pack(list_: npt.ArrayLike, width: int) -> npt.NDArray[np.int_]:
    r"""
    Packs a given bit array. Splits the bit array into groups of `width` bits and converts each group to its integer value.

    Parameters:
        list_: The input bit array.
        width: The width of each group.

    Returns:
        packed (Array1D[int]): The packed integer array.

    Examples:
        >>> komm.pack([0, 0, 0, 0, 1, 0, 1, 0, 1, 1], width=5)
        array([16, 26])

        >>> komm.pack([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], width=8)
        array([16, 26])
    """
    return np.apply_along_axis(binlist2int, 1, np.reshape(list_, shape=(-1, width)))


def unpack(list_: Iterable[int], width: int) -> npt.NDArray[np.int_]:
    r"""
    Unpacks a given integer array. Unpacks a given integer array by converting each integer to its bit array representation, using the specified `width` for each group, and concatenating the results.

    Parameters:
        list_: The input integer array.
        width: The width of each group.

    Returns:
        unpacked (Array1D[int]): The unpacked bit array.

    Examples:
        >>> komm.unpack([16, 26], width=5)
        array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])

        >>> komm.unpack([16, 26], width=8)
        array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    """
    return np.ravel([int2binlist(i, width=width) for i in list_])
