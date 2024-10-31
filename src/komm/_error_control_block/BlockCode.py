import itertools as it
from functools import cached_property

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .._algebra._util import null_matrix, right_inverse, rref
from .._util import binlist2int, int2binlist


@frozen(kw_only=True)
class BlockCode:
    r"""
    General binary linear block code. It is characterized by its *generator matrix* $G \in \mathbb{B}^{k \times n}$, and by its *check matrix* $H \in \mathbb{B}^{m \times n}$, which are related by $G H^\top = 0$. The parameters $n$, $k$, and $m$ are called the code *length*, *dimension*, and *redundancy*, respectively, and are related by $k + m = n$. For more details, see <cite>LC04, Ch. 3</cite>.

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
    _generator_matrix: npt.ArrayLike = field(
        default=None, repr=False, alias="generator_matrix"
    )
    _check_matrix: npt.ArrayLike = field(default=None, repr=False, alias="check_matrix")

    def __repr__(self) -> str:
        args = {}
        if self._generator_matrix is not None:
            args["generator_matrix"] = self.generator_matrix.tolist()
        if self._check_matrix is not None:
            args["check_matrix"] = self.check_matrix.tolist()
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in args.items())})"

    @cached_property
    def generator_matrix(self) -> np.ndarray:
        if self._generator_matrix is not None:
            return np.asarray(self._generator_matrix)
        return rref(null_matrix(self.check_matrix))

    @cached_property
    def check_matrix(self) -> np.ndarray:
        if self._check_matrix is not None:
            return np.asarray(self._check_matrix)
        return null_matrix(self.generator_matrix)

    @property
    def length(self) -> int:
        r"""
        The length $n$ of the code.
        """
        try:
            return self.generator_matrix.shape[1]
        except AttributeError:
            return self.check_matrix.shape[1]
        finally:
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
    def generator_matrix_right_inverse(self) -> np.ndarray:
        return right_inverse(self.generator_matrix)

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

    def enc_mapping(self, u: npt.ArrayLike) -> np.ndarray:
        r"""
        The encoding mapping $\Enc : \mathbb{B}^k \to \mathbb{B}^n$ of the code. This is a function that takes a message $u \in \mathbb{B}^k$ and returns the corresponding codeword $v \in \mathbb{B}^n$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> enc_mapping = code.enc_mapping
            >>> enc_mapping([1, 0, 1])
            array([1, 0, 1, 1, 0, 1])
        """
        v = np.dot(u, self.generator_matrix) % 2
        return v

    def inv_enc_mapping(self, v: npt.ArrayLike) -> np.ndarray:
        r"""
        The inverse encoding mapping $\Enc^{-1} : \mathbb{B}^n \to \mathbb{B}^k$ of the code. This is a function that takes a codeword $v \in \mathbb{B}^n$ and returns the corresponding message $u \in \mathbb{B}^k$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> inv_enc_mapping = code.inv_enc_mapping
            >>> inv_enc_mapping([1, 0, 1, 1, 0, 1])
            array([1, 0, 1])
        """
        v = np.asarray(v)
        if v.size != self.length:
            raise ValueError("The length of 'v' must be equal to the code length")
        s = self.chk_mapping(v)
        if not np.all(s == 0):
            raise ValueError("The input 'v' is not a valid codeword")
        u = np.dot(v, self.generator_matrix_right_inverse) % 2
        return u

    @cached_property
    def codewords(self) -> np.ndarray:
        r"""
        The codewords of the code. This is a $2^k \times n$ matrix whose rows are all the codewords. The codeword in row $i$ corresponds to the message whose binary representation (MSB in the right) is $i$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.codewords
            array([[0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1, 1],
                   [0, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0],
                   [1, 0, 1, 1, 0, 1],
                   [0, 1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 0, 0]])
        """
        k = self.dimension
        messages = np.array([int2binlist(i, width=k) for i in range(2**k)], dtype=int)
        return np.dot(messages, self.generator_matrix) % 2

    @cached_property
    def codeword_weight_distribution(self) -> np.ndarray:
        r"""
        The codeword weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of codewords of Hamming weight $w$, for $w \in [0 : n]$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.codeword_weight_distribution
            array([1, 0, 0, 4, 3, 0, 0])
        """
        return np.bincount(np.sum(self.codewords, axis=1), minlength=self.length + 1)

    def chk_mapping(self, r: npt.ArrayLike) -> np.ndarray:
        r"""
        The check mapping $\mathrm{Chk}: \mathbb{B}^n \to \mathbb{B}^m$ of the code. This is a function that takes a received word $r \in \mathbb{B}^n$ and returns the corresponding syndrome $s \in \mathbb{B}^m$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> chk_mapping = code.chk_mapping
            >>> chk_mapping([1, 0, 1, 1, 0, 1])
            array([0, 0, 0])
        """
        s = np.dot(r, self.check_matrix.T) % 2
        return s

    @cached_property
    def coset_leaders(self) -> np.ndarray:
        r"""
        The coset leaders of the code. This is a $2^m \times n$ matrix whose rows are all the coset leaders. The coset leader in row $i$ corresponds to the syndrome whose binary representation (MSB in the right) is $i$, and whose Hamming weight is minimal. This may be used as a LUT for syndrome-based decoding.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.coset_leaders
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0]])
        """
        m, n = self.redundancy, self.length
        coset_leaders = np.empty([2**m, n], dtype=int)
        taken = []
        for w in range(n + 1):
            for idx in it.combinations(range(n), w):
                e = np.zeros(n, dtype=int)
                e[list(idx)] = 1
                s = np.dot(e, self.check_matrix.T) % 2
                s_int = binlist2int(s)
                if s_int not in taken:
                    coset_leaders[s_int] = e
                    taken.append(s_int)
                if len(taken) == 2**m:
                    break
        return coset_leaders

    @cached_property
    def coset_leader_weight_distribution(self) -> np.ndarray:
        r"""
        The coset leader weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of coset leaders of weight $w$, for $w \in [0 : n]$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.coset_leader_weight_distribution
            array([1, 6, 1, 0, 0, 0, 0])
        """
        return np.bincount(
            np.sum(self.coset_leaders, axis=1), minlength=self.length + 1
        )

    @cached_property
    def minimum_distance(self) -> int:
        r"""
        The minimum distance $d$ of the code. This is equal to the minimum Hamming weight of the non-zero codewords.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.minimum_distance
            3
        """
        return np.flatnonzero(self.codeword_weight_distribution)[1]

    @cached_property
    def packing_radius(self) -> int:
        r"""
        The packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to $\lfloor (d - 1) / 2 \rfloor$.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.packing_radius
            1
        """
        return (self.minimum_distance - 1) // 2

    @cached_property
    def covering_radius(self) -> int:
        r"""
        The covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders.

        Examples:

            >>> code = komm.BlockCode(generator_matrix=[[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
            >>> code.covering_radius
            2
        """
        return np.flatnonzero(self.coset_leader_weight_distribution)[-1]

    @property
    def default_decoder(self) -> str:
        return (
            "exhaustive_search_hard"
            if self.dimension <= self.redundancy
            else "syndrome_table"
        )

    @classmethod
    def supported_decoders(cls) -> list[str]:
        return ["exhaustive_search_hard", "exhaustive_search_soft", "syndrome_table"]
