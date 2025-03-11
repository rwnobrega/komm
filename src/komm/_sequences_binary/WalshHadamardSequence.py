from typing import Literal

from .._util.bit_operations import bits_to_int, int_to_bits
from .BinarySequence import BinarySequence
from .sequences import hadamard_matrix


class WalshHadamardSequence(BinarySequence):
    r"""
    Walsh–Hadamard sequence. Consider the following recursive matrix construction:
    $$
        H_1 =
        \begin{bmatrix}
            +1
        \end{bmatrix}, \qquad
        H_{2^n} =
        \begin{bmatrix}
            H_{2^{n-1}} & H_{2^{n-1}} \\\\
            H_{2^{n-1}} & -H_{2^{n-1}}
        \end{bmatrix},
    $$
    for $n = 1, 2, \ldots$. For example, for $n = 3$,
    $$
        H_8 =
        \begin{bmatrix}
            +1 & +1 & +1 & +1 & +1 & +1 & +1 & +1 \\\\
            +1 & -1 & +1 & -1 & +1 & -1 & +1 & -1 \\\\
            +1 & +1 & -1 & -1 & +1 & +1 & -1 & -1 \\\\
            +1 & -1 & -1 & +1 & +1 & -1 & -1 & +1 \\\\
            +1 & +1 & +1 & +1 & -1 & -1 & -1 & -1 \\\\
            +1 & -1 & +1 & -1 & -1 & +1 & -1 & +1 \\\\
            +1 & +1 & -1 & -1 & -1 & -1 & +1 & +1 \\\\
            +1 & -1 & -1 & +1 & -1 & +1 & +1 & -1 \\\\
        \end{bmatrix}
    $$
    The above matrix is said to be in *natural ordering*. If the rows of the matrix are rearranged by first applying the bit-reversal permutation and then the Gray-code permutation, the following matrix is obtained:
    $$
        H_8^{\mathrm{s}} =
        \begin{bmatrix}
            +1 & +1 & +1 & +1 & +1 & +1 & +1 & +1 \\\\
            +1 & +1 & +1 & +1 & -1 & -1 & -1 & -1 \\\\
            +1 & +1 & -1 & -1 & -1 & -1 & +1 & +1 \\\\
            +1 & +1 & -1 & -1 & +1 & +1 & -1 & -1 \\\\
            +1 & -1 & -1 & +1 & +1 & -1 & -1 & +1 \\\\
            +1 & -1 & -1 & +1 & -1 & +1 & +1 & -1 \\\\
            +1 & -1 & +1 & -1 & -1 & +1 & -1 & +1 \\\\
            +1 & -1 & +1 & -1 & +1 & -1 & +1 & -1 \\\\
        \end{bmatrix}
    $$
    The above matrix is said to be in *sequency ordering*. It has the property that row $i$ has exactly $i$ sign changes.

    The Walsh–Hadamard sequence of *length* $L$ and *index* $i \in [0 : L)$ is a [binary sequence](/ref/BinarySequence) whose polar format is the $i$-th row of $H_L$, if assuming natural ordering, or $H_L^{\mathrm{s}}$, if assuming sequency ordering. Fore more details, see [Wikipedia: Hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix) and [Wikipedia: Walsh matrix](https://en.wikipedia.org/wiki/Walsh_matrix).

    Parameters:
        length: Length $L$ of the Walsh–Hadamard sequence. Must be a power of two.

        ordering: Ordering to be assumed. Should be one of `'natural'`, `'sequency'`, or `'dyadic'`. The default value is `'natural'`.

        index: Index of the Walsh–Hadamard sequence, with respect to the ordering assumed. Must be in the set $[0 : L)$. The default value is `0`.

    Examples:
        >>> walsh_hadamard = komm.WalshHadamardSequence(length=8, ordering='natural', index=5)
        >>> walsh_hadamard.polar_sequence
        array([ 1, -1,  1, -1, -1,  1, -1,  1])


        >>> walsh_hadamard = komm.WalshHadamardSequence(length=8, ordering='sequency', index=5)
        >>> walsh_hadamard.polar_sequence
        array([ 1, -1, -1,  1, -1,  1,  1, -1])

        >>> walsh_hadamard = komm.WalshHadamardSequence(length=8, ordering='dyadic', index=5)
        Traceback (most recent call last):
        ...
        NotImplementedError
    """

    def __init__(
        self,
        length: int,
        ordering: Literal["natural", "sequency", "dyadic"] = "natural",
        index: int = 0,
    ) -> None:
        if length & (length - 1):
            raise ValueError("'length' must be a power of two")
        if not 0 <= index < length:
            raise ValueError("'index' must be in [0 : length)")
        if ordering not in {"natural", "sequency", "dyadic"}:
            raise ValueError("'ordering' must be in {'natural', 'sequency', 'dyadic'}")

        if ordering == "natural":
            natural_index = index
        elif ordering == "sequency":
            width = (length - 1).bit_length()
            index_gray = index ^ (index >> 1)
            natural_index = bits_to_int(int_to_bits(index_gray, width)[::-1])
        elif ordering == "dyadic":
            raise NotImplementedError

        self.index = index
        self.ordering = ordering
        super().__init__(polar_sequence=hadamard_matrix(length)[natural_index])

    def __repr__(self) -> str:
        args = f"length={self.length}, ordering={self.ordering!r}, index={self.index}"
        return f"{self.__class__.__name__}({args})"
