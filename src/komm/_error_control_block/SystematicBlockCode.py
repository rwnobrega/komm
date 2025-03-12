from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._util.decorators import blockwise
from . import base


class SystematicBlockCode(base.BlockCode):
    r"""
    Systematic linear block code. A *systematic linear block code* is a [linear block code](/ref/BlockCode) in which the information bits can be found in predefined positions in the codeword, called the *information set* $\mathcal{K}$, which is a $k$-sublist of $[0 : n)$; the remaining positions are called the *parity set* $\mathcal{M}$, which is a $m$-sublist of $[0 : n)$. In this case, the generator matrix then has the property that the columns indexed by $\mathcal{K}$ are equal to $I_k$, and the columns indexed by $\mathcal{M}$ are equal to $P$. The check matrix has the property that the columns indexed by $\mathcal{M}$ are equal to $I_m$, and the columns indexed by $\mathcal{K}$ are equal to $P^\transpose$. The matrix $P \in \mathbb{B}^{k \times m}$ is called the *parity submatrix* of the code.

    The constructor expects the parity submatrix and the information set.

    Attributes:
        parity_submatrix: The parity submatrix $P$ the code, which is a $k \times m$ binary matrix.

        information_set: Either an array containing the indices of the information positions, which must be a $k$-sublist of $[0 : n)$, or one of the strings `'left'` or `'right'`. The default value is `'left'`.

    Examples:
        >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
        >>> (code.length, code.dimension, code.redundancy)
        (5, 2, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1],
               [0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[0, 1, 1, 0, 0],
               [1, 1, 0, 1, 0],
               [1, 0, 0, 0, 1]])

        >>> code = komm.SystematicBlockCode(
        ...     parity_submatrix=[[0, 1, 1], [1, 1, 0]],
        ...     information_set='right',
        ... )
        >>> (code.length, code.dimension, code.redundancy)
        (5, 2, 3)
        >>> code.generator_matrix
        array([[0, 1, 1, 1, 0],
               [1, 1, 0, 0, 1]])
        >>> code.check_matrix
        array([[1, 0, 0, 0, 1],
               [0, 1, 0, 1, 1],
               [0, 0, 1, 1, 0]])
    """

    def __init__(
        self,
        parity_submatrix: npt.ArrayLike,
        information_set: Literal["left", "right"] | npt.ArrayLike = "left",
    ):
        self.parity_submatrix = np.asarray(parity_submatrix)
        n, k, m = self.length, self.dimension, self.redundancy
        if isinstance(information_set, str):
            if information_set == "left":
                self.information_set = np.arange(k)
            elif information_set == "right":
                self.information_set = np.arange(m, n)
            else:
                raise ValueError(
                    "if string, 'information_set' must be 'left' or 'right'"
                )
            return
        self.information_set = np.asarray(information_set)
        if (
            self.information_set.size != k
            or self.information_set.min() < 0
            or self.information_set.max() > n
        ):
            raise ValueError("'information_set' must be a 'k'-sublist of 'range(n)'")

    def __repr__(self) -> str:
        args = ", ".join([
            f"parity_submatrix={self.parity_submatrix.tolist()}",
            f"information_set={self.information_set.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def parity_set(self) -> npt.NDArray[np.integer]:
        return np.setdiff1d(np.arange(self.length), self.information_set)

    @cached_property
    def length(self) -> int:
        r"""
        Examples:
           >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
           >>> code.length
           5
        """
        return self.dimension + self.redundancy

    @cached_property
    def dimension(self) -> int:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.dimension
            2
        """
        return self.parity_submatrix.shape[0]

    @cached_property
    def redundancy(self) -> int:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.redundancy
            3
        """
        return self.parity_submatrix.shape[1]

    @cached_property
    def rate(self) -> float:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.rate
            0.4
        """
        return super().rate

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.generator_matrix
            array([[1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0]])
        """
        k, n = self.dimension, self.length
        matrix = np.empty((k, n), dtype=int)
        matrix[:, self.information_set] = np.eye(k, dtype=int)
        matrix[:, self.parity_set] = self.parity_submatrix
        return matrix

    @cached_property
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        k, n = self.dimension, self.length
        matrix = np.zeros((n, k), dtype=int)
        matrix[self.information_set] = np.eye(k, dtype=int)
        return matrix

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.check_matrix
            array([[0, 1, 1, 0, 0],
                   [1, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1]])
        """
        m, n = self.redundancy, self.length
        matrix = np.empty((m, n), dtype=int)
        matrix[:, self.information_set] = self.parity_submatrix.T
        matrix[:, self.parity_set] = np.eye(m, dtype=int)
        return matrix

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.generator_matrix
            array([[1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0]])
            >>> code.encode([0, 0])  # Sequence with single message
            array([0, 0, 0, 0, 0])
            >>> code.encode([0, 0, 1, 1])  # Sequence with two messages
            array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
            >>> code.encode([[0, 0],  # 2D array of single messages
            ...              [1, 1]])
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 1]])
            >>> code.encode([[0, 0, 1, 1],  # 2D array of two messages
            ...              [1, 1, 1, 0]])
            array([[0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]])
        """

        @blockwise(self.dimension)
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            v = np.empty(u.shape[:-1] + (self.length,), dtype=int)
            v[..., self.information_set] = u
            v[..., self.parity_set] = u @ self.parity_submatrix % 2
            return v

        return encode(input)

    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        def project(v: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            u = v[..., self.information_set]
            return u

        return project(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.inverse_encode([0, 0, 0, 0, 0])  # Sequence with single codeword
            array([0, 0])
            >>> code.inverse_encode([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])  # Sequence with two codewords
            array([0, 0, 1, 1])
            >>> code.inverse_encode([[0, 0, 0, 0, 0],  # 2D array of single codewords
            ...                      [1, 1, 1, 0, 1]])
            array([[0, 0],
                   [1, 1]])
            >>> code.inverse_encode([[0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # 2D array of two codewords
            ...                      [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]])
            array([[0, 0, 1, 1],
                   [1, 1, 1, 0]])
        """
        return super().inverse_encode(input)

    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.check([1, 1, 1, 0, 1])  # Sequence with single received word
            array([0, 0, 0])
            >>> code.check([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])  # Sequence with two received words
            array([0, 0, 0, 0, 1, 0])
            >>> code.check([[1, 1, 1, 0, 1],  # 2D array of single received words
            ...             [1, 1, 1, 1, 1]])
            array([[0, 0, 0],
                   [0, 1, 0]])
            >>> code.check([[1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # 2D array of two received words
            ...             [1, 1, 1, 1, 1, 0, 0, 0, 1, 1]])
            array([[0, 0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 1, 1]])
        """

        @blockwise(self.length)
        def check(r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            r_inf = r[..., self.information_set]
            r_par = r[..., self.parity_set]
            s = (r_inf @ self.parity_submatrix + r_par) % 2
            return s

        return check(input)

    @cache
    def codewords(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.codewords()
            array([[0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 0, 1]])
        """
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.codeword_weight_distribution()
            array([1, 0, 0, 2, 1, 0])
        """
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.minimum_distance()
            3
        """
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.coset_leaders()
            array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0]])
        """
        return super().coset_leaders()

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.coset_leader_weight_distribution()
            array([1, 5, 2, 0, 0, 0])
        """
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.packing_radius()
            1
        """
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]])
            >>> code.covering_radius()
            2
        """
        return super().covering_radius()
