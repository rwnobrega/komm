import numpy as np


def _binlist2int(binlist):
    return sum(1 << i for (i, b) in enumerate(binlist) if b != 0)


def binlist2int(binlist):
    r"""
    Converts a bit array to its integer representation.

    Parameters:

        binlist (List[int] | Array1D[int]): A list or array of $0$'s and $1$'s whose $i$-th element stands for the coefficient of $2^i$ in the binary representation of the output integer.

    Returns:

        integer (int): The integer representation of the input bit array.

    Examples:

        >>> komm.binlist2int([0, 1, 0, 1, 1])
        26

        >>> komm.binlist2int([0, 1, 0, 1, 1, 0, 0, 0])
        26
    """
    return _binlist2int(binlist)


def _int2binlist(integer, width=None):
    if width is None:
        width = max(integer.bit_length(), 1)
    return [(integer >> i) & 1 for i in range(width)]


def int2binlist(integer, width=None):
    r"""
    Converts an integer to its bit array representation.

    Parameters:

        integer (int): The input integer. May be any nonnegative integer.

        width (Optional[int]): If this parameter is specified, the output will be filled with zeros on the right so that its length will be the specified value.

    Returns:

        binlist (Array1D[int]): An array of $0$'s and $1$'s whose $i$-th element stands for the coefficient of $2^i$ in the binary representation of the input integer.

    Examples:

        >>> komm.int2binlist(26)
        array([0, 1, 0, 1, 1])

        >>> komm.int2binlist(26, width=8)
        array([0, 1, 0, 1, 1, 0, 0, 0])
    """
    return np.array(_int2binlist(integer, width))


def _pack(list_, width):
    return np.apply_along_axis(_binlist2int, 1, np.reshape(list_, newshape=(-1, width)))


def pack(list_, width):
    r"""
    Packs a given integer array.
    """
    return _pack(list_, width)


def _unpack(list_, width):
    return np.ravel([_int2binlist(i, width=width) for i in list_])


def unpack(list_, width):
    r"""
    Unpacks a given bit array.
    """
    return _unpack(list_, width)
