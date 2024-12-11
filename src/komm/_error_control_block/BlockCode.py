from functools import cache, cached_property
from typing import final

import numpy as np
import numpy.typing as npt
from attrs import field, frozen
from tqdm import tqdm

from .. import abc
from .._util.matrices import null_matrix, pseudo_inverse, rref
from .SlepianArray import SlepianArray


@frozen(kw_only=True, eq=False)
class BlockCode(abc.BlockCode):
    r"""
    General binary linear block code. It is characterized by its *generator matrix* $G \in \mathbb{B}^{k \times n}$, and by its *check matrix* $H \in \mathbb{B}^{m \times n}$, which are related by $G H^\transpose = 0$. The parameters $n$, $k$, and $m$ are called the code *length*, *dimension*, and *redundancy*, respectively, and are related by $k + m = n$. For more details, see <cite>LC04, Ch. 3</cite>.

    The constructor expects either the generator matrix or the check matrix.

    Parameters:
        generator_matrix (Array2D[int]): The generator matrix $G$ of the code, which is a $k \times n$ binary matrix.
        check_matrix (Array2D[int]): The check matrix $H$ of the code, which is a $m \times n$ binary matrix.

    Examples:
        >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[0, 1, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0],
               [1, 1, 0, 0, 0, 1]])

        >>> code = komm.BlockCode(check_matrix=[[0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[0, 1, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0],
               [1, 1, 0, 0, 0, 1]])
    """

    _generator_matrix: npt.ArrayLike | None = field(
        default=None, repr=False, alias="generator_matrix"
    )
    _check_matrix: npt.ArrayLike | None = field(
        default=None, repr=False, alias="check_matrix"
    )

    def __repr__(self) -> str:
        if self._generator_matrix is not None:
            args = f"generator_matrix={self.generator_matrix.tolist()}"
        else:  # self._check_matrix is not None
            args = f"check_matrix={self.check_matrix.tolist()}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        if self._generator_matrix is not None:
            return np.asarray(self._generator_matrix)
        return rref(null_matrix(self.check_matrix))

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        if self._check_matrix is not None:
            return np.asarray(self._check_matrix)
        return null_matrix(self.generator_matrix)

    @property
    def length(self) -> int:
        r"""
        The length $n$ of the code.
        """
        if self._generator_matrix is not None:
            return self.generator_matrix.shape[1]
        if self._check_matrix is not None:
            return self.check_matrix.shape[1]
        return self.dimension + self.redundancy

    @property
    def dimension(self) -> int:
        r"""
        The dimension $k$ of the code.
        """
        try:
            return self.generator_matrix.shape[0]
        except AttributeError:
            return self.length - self.redundancy

    @property
    def redundancy(self) -> int:
        r"""
        The redundancy $m$ of the code.
        """
        try:
            return self.check_matrix.shape[0]
        except AttributeError:
            return self.length - self.dimension

    @cached_property
    def _generator_matrix_pseudo_inverse(self) -> npt.NDArray[np.integer]:
        return pseudo_inverse(self.generator_matrix)

    @property
    def rate(self) -> float:
        r"""
        The rate $R = k/n$ of the code.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.rate
            0.5
        """
        return self.dimension / self.length

    @final
    def encode(self, u: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the encoding mapping $\Enc : \mathbb{B}^k \to \mathbb{B}^n$ of the code. This method takes one or more messages $u \in \mathbb{B}^k$ and returns the corresponding codeword(s) $v \in \mathcal{C}$.

        Parameters:
            u: The input message(s). Can be a single message of length $k$ or a multidimensional array where the last dimension has length $k$.

        Returns:
            v: The output codeword(s). Has the same shape as the input, with the last dimension expanded from $k$ to $n$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.encode([1, 0, 1])  # Single message
            array([1, 0, 1, 1, 0, 1])
            >>> code.encode([[1, 0, 1], [0, 1, 0]])  # Multiple messages
            array([[1, 0, 1, 1, 0, 1],
                   [0, 1, 0, 1, 0, 1]])
        """
        u = np.asarray(u)
        if u.shape[-1] != self.dimension:
            raise ValueError("last dimension of 'u' should be the code dimension")
        return self._encode(u)

    def _encode(self, u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        v = u @ self.generator_matrix % 2
        return v

    @final
    def inverse_encode(self, v: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the inverse encoding mapping $\Enc^{-1} : \mathbb{B}^n \to \mathbb{B}^k$ of the code. This is a function that takes one or more codewords $v \in \mathcal{C}$ and returns the corresponding message(s) $u \in \mathbb{B}^k$.

        Parameters:
            v: The input codeword(s). Can be a single codeword of length $n$ or a multidimensional array where the last dimension has length $n$.

        Returns:
            u: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.inverse_encode([1, 0, 1, 1, 0, 1])  # Single codeword
            array([1, 0, 1])
            >>> code.inverse_encode([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1]])  # Multiple codewords
            array([[1, 0, 1],
                   [0, 1, 0]])
        """
        v = np.asarray(v)
        if v.shape[-1] != self.length:
            raise ValueError("last dimension of 'v' should be the code length")
        return self._inverse_encode(v)

    def _inverse_encode(self, v: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        s = self._check(v)
        if not np.all(s == 0):
            raise ValueError("one or more inputs in 'v' are not valid codewords")
        u = v @ self._generator_matrix_pseudo_inverse % 2
        return u

    @final
    def check(self, r: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the check mapping $\mathrm{Chk}: \mathbb{B}^n \to \mathbb{B}^m$ of the code. This is a function that takes one or more received words $r \in \mathbb{B}^n$ and returns the corresponding syndrome(s) $s \in \mathbb{B}^m$.

        Parameters:
            r: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

        Returns:
            s: The output syndrome(s). Has the same shape as the input, with the last dimension reduced from $n$ to $m$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.check([1, 0, 1, 1, 0, 1])  # Single received word
            array([0, 0, 0])
            >>> code.check([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0]])  # Multiple received words
            array([[0, 0, 0],
                   [0, 0, 1]])
        """
        r = np.asarray(r)
        if r.shape[-1] != self.length:
            raise ValueError("last dimension of 'r' should be the code length")
        return self._check(r)

    def _check(self, r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        s = r @ self.check_matrix.T % 2
        return s

    @cache
    def codewords(self, _batch_size: int = 1024) -> npt.NDArray[np.integer]:
        r"""
        Returns the codewords of the code. This is a $2^k \times n$ matrix whose rows are all the codewords. The codeword in row $i$ corresponds to the message obtained by expressing $i$ in binary with $k$ bits (MSB in the right).

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.codewords()
            array([[0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1, 1],
                   [0, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0],
                   [1, 0, 1, 1, 0, 1],
                   [0, 1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 0, 0]])
        """
        k, n = self.dimension, self.length
        codewords = np.empty((2**k, n), dtype=int)
        for i in tqdm(
            range(0, 2**k, _batch_size), desc="Generating codewords", delay=2.5
        ):
            batch_end = min(i + _batch_size, 2**k)
            js = np.arange(i, batch_end, dtype=np.uint64).reshape(-1, 1).view(np.uint8)
            messages_batch = np.unpackbits(js, axis=1, count=k, bitorder="little")
            codewords[i:batch_end] = self.encode(messages_batch)
        return codewords

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the codeword weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of codewords of Hamming weight $w$, for $w \in [0 : n]$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.codeword_weight_distribution()
            array([1, 0, 0, 4, 3, 0, 0])
        """
        return np.bincount(np.sum(self.codewords(), axis=1), minlength=self.length + 1)

    @cache
    def minimum_distance(self) -> int:
        r"""
        Returns the minimum distance $d$ of the code. This is equal to the minimum Hamming weight of the non-zero codewords.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.minimum_distance()
            3
        """
        return int(np.flatnonzero(self.codeword_weight_distribution())[1])

    @cache
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the coset leaders of the code. This is a $2^m \times n$ matrix whose rows are all the coset leaders. The coset leader in row $i$ corresponds to the syndrome obtained by expressing $i$ in binary with $m$ bits (MSB in the right), and whose Hamming weight is minimal. This may be used as a LUT for syndrome-based decoding.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.coset_leaders()
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0]])
        """
        sa = SlepianArray(self)
        return sa.col(0)

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the coset leader weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of coset leaders of weight $w$, for $w \in [0 : n]$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.coset_leader_weight_distribution()
            array([1, 6, 1, 0, 0, 0, 0])
        """
        return np.bincount(
            np.sum(self.coset_leaders(), axis=1), minlength=self.length + 1
        )

    @cache
    def packing_radius(self) -> int:
        r"""
        Returns the packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to $\lfloor (d - 1) / 2 \rfloor$.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.packing_radius()
            1
        """
        return (self.minimum_distance() - 1) // 2

    @cache
    def covering_radius(self) -> int:
        r"""
        Returns the covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders.

        Examples:
            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.covering_radius()
            2
        """
        return int(np.flatnonzero(self.coset_leader_weight_distribution())[-1])
