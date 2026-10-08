from typing import Literal

from .._util.bit_operations import from_binary, to_binary
from .._util.validators import validate_choice, validate_integer
from .BinarySequence import BinarySequence
from .sequences import hadamard_matrix


class WalshHadamardSequence(BinarySequence):
    r"""
    Walsh–Hadamard sequence. The Walsh–Hadamard sequence of *length* $L$ and *index* $i \in [0 : L)$ is a [binary sequence](/ref/BinarySequence) whose polar format is the $i$-th row of $H_L$, $H_L^{\mathrm{d}}$, or $H_L^{\mathrm{s}}$, for natural, dyadic, or sequency *ordering*, respectively. These are $L \times L$ matrices with the same rows, in different orders.

    In *natural ordering* (also known as *Hadamard ordering*), the matrix $H_L$ is defined recursively by
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
    for $n = 1, 2, \ldots$. In *dyadic ordering* (also known as *Paley ordering*), the matrix $H_L^{\mathrm{d}}$ is obtained by applying the bit-reversal permutation to the rows of $H_L$. In *sequency ordering* (also known as *Walsh ordering*), the matrix $H_L^{\mathrm{s}}$ is obtained by applying the Gray-code permutation to the rows of $H_L^{\mathrm{d}}$; its row $i$ has exactly $i$ sign changes. For example, for $L = 8$,
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
    $$
        H_8^{\mathrm{d}} =
        \begin{bmatrix}
            +1 & +1 & +1 & +1 & +1 & +1 & +1 & +1 \\\\
            +1 & +1 & +1 & +1 & -1 & -1 & -1 & -1 \\\\
            +1 & +1 & -1 & -1 & +1 & +1 & -1 & -1 \\\\
            +1 & +1 & -1 & -1 & -1 & -1 & +1 & +1 \\\\
            +1 & -1 & +1 & -1 & +1 & -1 & +1 & -1 \\\\
            +1 & -1 & +1 & -1 & -1 & +1 & -1 & +1 \\\\
            +1 & -1 & -1 & +1 & +1 & -1 & -1 & +1 \\\\
            +1 & -1 & -1 & +1 & -1 & +1 & +1 & -1 \\\\
        \end{bmatrix}
    $$
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

    For more details, see [Wikipedia: Hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix) and [Wikipedia: Walsh matrix](https://en.wikipedia.org/wiki/Walsh_matrix).

    Parameters:
        length: Length $L$ of the Walsh–Hadamard sequence. Must be a power of two.

        ordering: Ordering to be assumed. Should be one of `'natural'`, `'sequency'`, or `'dyadic'`. The default value is `'natural'`.

        index: Index of the Walsh–Hadamard sequence, with respect to the ordering assumed. Must be in the set $[0 : L)$. The default value is `0`.

    Examples:
        >>> walsh_hadamard = komm.WalshHadamardSequence(8, ordering='natural', index=6)
        >>> walsh_hadamard.polar_sequence
        array([ 1,  1, -1, -1, -1, -1,  1,  1])

        >>> walsh_hadamard = komm.WalshHadamardSequence(8, ordering='sequency', index=6)
        >>> walsh_hadamard.polar_sequence
        array([ 1, -1,  1, -1, -1,  1, -1,  1])

        >>> walsh_hadamard = komm.WalshHadamardSequence(8, ordering='dyadic', index=6)
        >>> walsh_hadamard.polar_sequence
        array([ 1, -1, -1,  1,  1, -1, -1,  1])
    """

    def __init__(
        self,
        length: int,
        ordering: Literal["natural", "sequency", "dyadic"] = "natural",
        index: int = 0,
    ) -> None:
        length = validate_integer(length, "length", low=1)
        if length & (length - 1):
            raise ValueError("'length' must be a power of two")
        rule = "0 <= index < length"
        index = validate_integer(index, "index", low=0, high=length, rule=rule)
        orderings = ("natural", "sequency", "dyadic")
        ordering = validate_choice(ordering, "ordering", orderings)

        if ordering == "natural":
            natural_index = index
        elif ordering == "sequency":
            index_gray = index ^ (index >> 1)
            width = (length - 1).bit_length()
            natural_index = from_binary(to_binary(index_gray, width, "MSB-first"))
        elif ordering == "dyadic":
            width = (length - 1).bit_length()
            natural_index = from_binary(to_binary(index, width, "MSB-first"))

        self.index = index
        self.ordering = ordering
        super().__init__(polar_sequence=hadamard_matrix(length)[natural_index])

    def __repr__(self) -> str:
        args = f"length={self.length}, ordering={self.ordering!r}, index={self.index}"
        return f"{self.__class__.__name__}({args})"
