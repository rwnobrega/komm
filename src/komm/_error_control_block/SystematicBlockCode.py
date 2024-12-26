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

        >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]], information_set='right')
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

    @property
    def length(self) -> int:
        return self.dimension + self.redundancy

    @property
    def dimension(self) -> int:
        return self.parity_submatrix.shape[0]

    @property
    def redundancy(self) -> int:
        return self.parity_submatrix.shape[1]

    @property
    def rate(self) -> float:
        return super().rate

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
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
        m, n = self.redundancy, self.length
        matrix = np.empty((m, n), dtype=int)
        matrix[:, self.information_set] = self.parity_submatrix.T
        matrix[:, self.parity_set] = np.eye(m, dtype=int)
        return matrix

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.dimension)
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            v = np.empty(u.shape[:-1] + (self.length,), dtype=int)
            v[..., self.information_set] = u
            v[..., self.parity_set] = u @ self.parity_submatrix % 2
            return v

        return encode(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        s = self.check(input)
        if not np.all(s == 0):
            raise ValueError("one or more inputs in 'v' are not valid codewords")

        @blockwise(self.length)
        def inverse_encode(v: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            u = v[..., self.information_set]
            return u

        return inverse_encode(input)

    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        def check(r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            r_inf = r[..., self.information_set]
            r_par = r[..., self.parity_set]
            s = (r_inf @ self.parity_submatrix + r_par) % 2
            return s

        return check(input)

    @cache
    def codewords(self) -> npt.NDArray[np.integer]:
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        return super().coset_leaders()

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        return super().covering_radius()
