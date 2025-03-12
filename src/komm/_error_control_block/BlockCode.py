from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .._util.matrices import null_matrix, pseudo_inverse, rref
from . import base


class BlockCode(base.BlockCode):
    r"""
    General binary linear block code. It is characterized by its *generator matrix* $G \in \mathbb{B}^{k \times n}$, and by its *check matrix* $H \in \mathbb{B}^{m \times n}$, which are related by $G H^\transpose = 0$. The parameters $n$, $k$, and $m$ are called the code *length*, *dimension*, and *redundancy*, respectively, and are related by $k + m = n$. For more details, see <cite>LC04, Ch. 3</cite>.

    The constructor expects either the generator matrix or the check matrix.

    Attributes:
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
        if generator_matrix is not None and check_matrix is not None:
            raise ValueError(
                "only one of 'generator_matrix' or 'check_matrix' must be provided"
            )
        if generator_matrix is not None:
            self._generator_matrix = np.asarray(generator_matrix)
            self._check_matrix = None
            self._dimension, self._length = self._generator_matrix.shape
            self._redundancy = self._length - self._dimension
        else:  # check_matrix is not None
            self._generator_matrix = None
            self._check_matrix = np.asarray(check_matrix)
            self._redundancy, self._length = self._check_matrix.shape
            self._dimension = self._length - self._redundancy

    def __repr__(self) -> str:
        if self._generator_matrix is not None:
            args = f"generator_matrix={self.generator_matrix.tolist()}"
        else:  # self._check_matrix is not None
            args = f"check_matrix={self.check_matrix.tolist()}"
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
    def generator_matrix(self) -> npt.NDArray[np.integer]:
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
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        return pseudo_inverse(self.generator_matrix)

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
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

    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().project_word(input)

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
    def codewords(self) -> npt.NDArray[np.integer]:
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
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
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
    def coset_leaders(self) -> npt.NDArray[np.integer]:
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
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
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
