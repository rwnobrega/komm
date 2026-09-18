from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.matrices import null_matrix, rank, rref
from ..types import Array1D, Array2D


class BlockCode(abc.BlockCode):
    r"""
    General binary linear block code. It is characterized by its *generator matrix* $G \in \mathbb{B}^{k \times n}$, and by its *check matrix* $H \in \mathbb{B}^{m \times n}$, which are related by $G H^\transpose = 0$. The parameters $n$, $k$, and $m$ are called the code *length*, *dimension*, and *redundancy*, respectively, and are related by $k + m = n$. For more details, see <cite>LC04, Ch. 3</cite>.

    The constructor expects the generator matrix, the check matrix, or both. If both are provided, they must satisfy $G H^\transpose = 0$, and each one is kept as given, rather than derived from the other.

    Parameters:
        generator_matrix: The generator matrix $G$ of the code, which is a $k \times n$ binary matrix.

        check_matrix: The check matrix $H$ of the code, which is a $m \times n$ binary matrix.

    Examples:
        >>> code = komm.BlockCode(generator_matrix=[
        ...     [1, 0, 0, 1, 1],
        ...     [0, 1, 1, 1, 0],
        ... ])
        >>> (code.length, code.dimension, code.redundancy)
        (5, 2, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1],
               [0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[0, 1, 1, 0, 0],
               [1, 1, 0, 1, 0],
               [1, 0, 0, 0, 1]])

        >>> code = komm.BlockCode(check_matrix=[
        ...     [0, 1, 1, 0, 0],
        ...     [1, 1, 0, 1, 0],
        ...     [1, 0, 0, 0, 1],
        ... ])
        >>> (code.length, code.dimension, code.redundancy)
        (5, 2, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1],
               [0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[0, 1, 1, 0, 0],
               [1, 1, 0, 1, 0],
               [1, 0, 0, 0, 1]])

        >>> code = komm.BlockCode(
        ...     generator_matrix=[
        ...         [1, 0, 0, 1, 1],
        ...         [0, 1, 1, 1, 0],
        ...     ],
        ...     check_matrix=[
        ...         [1, 0, 1, 1, 0],
        ...         [1, 1, 0, 1, 0],
        ...         [1, 0, 0, 0, 1],
        ...     ],
        ... )
        >>> (code.length, code.dimension, code.redundancy)
        (5, 2, 3)
        >>> code.generator_matrix
        array([[1, 0, 0, 1, 1],
               [0, 1, 1, 1, 0]])
        >>> code.check_matrix
        array([[1, 0, 1, 1, 0],
               [1, 1, 0, 1, 0],
               [1, 0, 0, 0, 1]])
    """

    def __init__(
        self,
        generator_matrix: npt.ArrayLike | None = None,
        check_matrix: npt.ArrayLike | None = None,
    ):
        if generator_matrix is None and check_matrix is None:
            raise ValueError(
                "either 'generator_matrix' or 'check_matrix' must be provided"
            )
        G = None if generator_matrix is None else np.asarray(generator_matrix)
        H = None if check_matrix is None else np.asarray(check_matrix)
        if G is not None and H is not None:
            if G.shape[1] != H.shape[1] or G.shape[0] + H.shape[0] != G.shape[1]:
                raise ValueError("matrices must have compatible shapes")
            if np.any(G @ H.T % 2):
                raise ValueError("matrices must satisfy 'G H^T = 0'")
        if G is not None and rank(G) < G.shape[0]:
            raise ValueError("'generator_matrix' must have full row rank")
        if H is not None and rank(H) < H.shape[0]:
            raise ValueError("'check_matrix' must have full row rank")
        if G is not None:
            self._dimension, self._length = G.shape
            self._redundancy = self._length - self._dimension
        elif H is not None:
            self._redundancy, self._length = H.shape
            self._dimension = self._length - self._redundancy
        self._generator_matrix = G
        self._check_matrix = H

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._generator_matrix is not None:
            parts.append(f"generator_matrix={self._generator_matrix.tolist()}")
        if self._check_matrix is not None:
            parts.append(f"check_matrix={self._check_matrix.tolist()}")
        args = ", ".join(parts)
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def length(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.length
            5
        """
        return self._length

    @cached_property
    def dimension(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.dimension
            2
        """
        return self._dimension

    @cached_property
    def redundancy(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.redundancy
            3
        """
        return self._redundancy

    @cached_property
    def rate(self) -> float:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.rate
            0.4
        """
        return super().rate

    @cached_property
    def generator_matrix(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(check_matrix=[
            ...     [0, 1, 1, 0, 0],
            ...     [1, 1, 0, 1, 0],
            ...     [1, 0, 0, 0, 1],
            ... ])
            >>> code.generator_matrix
            array([[1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0]])
        """
        if self._generator_matrix is not None:
            return self._generator_matrix
        return rref(null_matrix(self.check_matrix))

    @cached_property
    def check_matrix(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.check_matrix
            array([[0, 1, 1, 0, 0],
                   [1, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1]])
        """
        if self._check_matrix is not None:
            return self._check_matrix
        return null_matrix(self.generator_matrix)

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
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
        return super().encode(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
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
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
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
        return super().check(input)

    @cache
    def codewords(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.codewords()
            array([[0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 1],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 0, 1]])
        """
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.codeword_weight_distribution()
            array([1, 0, 0, 2, 1, 0])
        """
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.minimum_distance()
            3
        """
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
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
    def coset_leader_weight_distribution(self) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.coset_leader_weight_distribution()
            array([1, 5, 2, 0, 0, 0])
        """
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.packing_radius()
            1
        """
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.BlockCode(generator_matrix=[
            ...     [1, 0, 0, 1, 1],
            ...     [0, 1, 1, 1, 0],
            ... ])
            >>> code.covering_radius()
            2
        """
        return super().covering_radius()
