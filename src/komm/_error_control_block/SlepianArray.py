from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._util.bit_operations import int_to_bits
from . import base


@dataclass
class SlepianArray:
    r"""
    Slepian array (standard array) for a [linear block code](/ref/BlockCode). It is a table with $2^m$ rows and $2^k$ columns, where $m$ is the redundancy, and $k$ is the dimension of the code. Each row corresponds to a _coset_ of the group of codewords, in which:

    - The first row is the group of codewords itself.

    - The first column contains _coset leaders_ (i.e., elements of minimal weight in its coset).

    In this implementation:

    - A row's index $i$ corresponds to the $m$-bit syndrome obtained by expressing $i$ in binary (MSB on the right).

    - A column's index $j$ corresponds to the $k$-bit message obtained by expressing $j$ in binary (MSB on the right).

    For more details, see <cite>LC04, Sec. 3.5</cite>.

    Attributes:
        code: The linear block code for which the Slepian array is generated.
    """

    code: base.BlockCode

    def __post_init__(self) -> None:
        self._leaders = self.code.coset_leaders()

    def entry(self, i: int, j: int) -> npt.NDArray[np.integer]:
        r"""
        The entry at the $i$-th row and $j$-th column of the Slepian array.

        Parameters:
            i: The index of the row.
            j: The index of the column.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> sa = komm.SlepianArray(code)
            >>> binlist2str = lambda binlist: "".join(str(bit) for bit in binlist)
            >>> m, k = code.redundancy, code.dimension
            >>> for i in range(2**m):
            ...     for j in range(2**k):
            ...         print(binlist2str(sa.entry(i, j)), end=" ")
            ...     print()
            000000 100011 010101 110110 001110 101101 011011 111000
            000100 100111 010001 110010 001010 101001 011111 111100
            000010 100001 010111 110100 001100 101111 011001 111010
            001000 101011 011101 111110 000110 100101 010011 110000
            000001 100010 010100 110111 001111 101100 011010 111001
            010000 110011 000101 100110 011110 111101 001011 101000
            100000 000011 110101 010110 101110 001101 111011 011000
            100100 000111 110001 010010 101010 001001 111111 011100
        """
        k = self.code.dimension
        leader = self._leaders[i]
        message = int_to_bits(j, k)
        codeword = self.code.encode(message)
        return np.array(leader + codeword) % 2

    def row(self, i: int) -> npt.NDArray[np.integer]:
        r"""
        The $i$-th row of the Slepian array.

        Parameters:
            i: The index of the row.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> sa = komm.SlepianArray(code)
            >>> sa.row(0)  # The codewords
            array([[0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1, 1],
                   [0, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0],
                   [1, 0, 1, 1, 0, 1],
                   [0, 1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 0, 0]])
        """
        k = self.code.dimension
        return np.array([self.entry(i, j) for j in range(2**k)])

    def col(self, j: int) -> npt.NDArray[np.integer]:
        r"""
        The $j$-th column of the Slepian array.

        Parameters:
            j: The index of the column.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> sa = komm.SlepianArray(code)
            >>> sa.col(0)  # The coset leaders
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0]])
        """
        m = self.code.redundancy
        return np.array([self.entry(i, j) for i in range(2**m)])
