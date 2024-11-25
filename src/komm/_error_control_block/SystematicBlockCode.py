from functools import cached_property

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .._types import ArrayIntLike
from .BlockCode import BlockCode


@frozen(kw_only=True, eq=False)
class SystematicBlockCode(BlockCode):
    r"""
    Systematic linear block code. A *systematic linear block code* is a [linear block code](/ref/BlockCode) in which the information bits can be found in predefined positions in the codeword, called the *information set* $\mathcal{K}$, which is a $k$-sublist of $[0 : n)$; the remaining positions are called the *parity set* $\mathcal{M}$, which is a $m$-sublist of $[0 : n)$. In this case, the generator matrix then has the property that the columns indexed by $\mathcal{K}$ are equal to $I_k$, and the columns indexed by $\mathcal{M}$ are equal to $P$. The check matrix has the property that the columns indexed by $\mathcal{M}$ are equal to $I_m$, and the columns indexed by $\mathcal{K}$ are equal to $P^\transpose$. The matrix $P \in \mathbb{B}^{k \times m}$ is called the *parity submatrix* of the code.

    The constructor expects the parity submatrix the information set.

    Parameters:
        parity_submatrix (Array2D[int]): The parity submatrix $P$ the code, which is a $k \times m$ binary matrix.

        information_set (Optional[Array1D[int] | str]): Either an array containing the indices of the information positions, which must be a $k$-sublist of $[0 : n)$, or one of the strings `'left'` or `'right'`. The default value is `'left'`.

    Examples:
        >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
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

        >>> code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], information_set='right')
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[0, 1, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0],
               [1, 1, 0, 0, 0, 1]])
        >>> code.check_matrix
        array([[1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 1, 0]])
    """

    _parity_submatrix: npt.ArrayLike = field(
        default=None, repr=False, alias="parity_submatrix"
    )
    _information_set: npt.ArrayLike | str = field(
        default="left", repr=False, alias="information_set"
    )

    def __repr__(self) -> str:
        args_str = f"parity_submatrix={self.parity_submatrix.tolist()}"
        args_str += f", information_set={self.information_set.tolist()}"
        return f"{self.__class__.__name__}({args_str})"

    @cached_property
    def parity_submatrix(self) -> npt.NDArray[np.int_]:
        return np.asarray(self._parity_submatrix)

    @cached_property
    def information_set(self) -> npt.NDArray[np.int_]:
        n, k, m = self.length, self.dimension, self.redundancy
        if self._information_set == "left":
            information_set = range(k)
        elif self._information_set == "right":
            information_set = range(m, n)
        else:
            information_set = self._information_set
        try:
            information_set = np.asarray(information_set)
        except TypeError:
            raise ValueError(
                "'information_set' must be either 'left', 'right', or an array of int"
            )
        if (
            information_set.size != k
            or information_set.min() < 0
            or information_set.max() > n
        ):
            raise ValueError("'information_set' must be a 'k'-sublist of 'range(n)'")
        return information_set

    @cached_property
    def parity_set(self) -> npt.NDArray[np.int_]:
        return np.setdiff1d(np.arange(self.length), self.information_set)

    @property
    def dimension(self) -> int:
        return self.parity_submatrix.shape[0]

    @property
    def redundancy(self) -> int:
        return self.parity_submatrix.shape[1]

    @property
    def length(self) -> int:
        return self.dimension + self.redundancy

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.int_]:
        k, n = self.dimension, self.length
        generator_matrix = np.empty((k, n), dtype=int)
        generator_matrix[:, self.information_set] = np.eye(k, dtype=int)
        generator_matrix[:, self.parity_set] = self.parity_submatrix
        return generator_matrix

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.int_]:
        m, n = self.redundancy, self.length
        check_matrix = np.empty((m, n), dtype=int)
        check_matrix[:, self.information_set] = self.parity_submatrix.T
        check_matrix[:, self.parity_set] = np.eye(m, dtype=int)
        return check_matrix

    def enc_mapping(self, u: ArrayIntLike) -> npt.NDArray[np.int_]:
        v = np.empty(self.length, dtype=int)
        v[self.information_set] = u
        v[self.parity_set] = np.dot(u, self.parity_submatrix) % 2
        return v

    def inv_enc_mapping(self, v: ArrayIntLike) -> npt.NDArray[np.int_]:
        v = np.asarray(v)
        if v.size != self.length:
            raise ValueError("length of 'v' must be equal to the code length")
        s = self.chk_mapping(v)
        if not np.all(s == 0):
            raise ValueError("input 'v' is not a valid codeword")
        u = np.take(v, self.information_set)
        return u
