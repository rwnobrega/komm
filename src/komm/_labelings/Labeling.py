import numpy as np
import numpy.typing as npt

from .. import abc


class Labeling(abc.Labeling):
    r"""
    General binary labeling. It is defined by a bijective mapping from $[0:2^m)$ to $\mathbb{B}^m$ or, equivalently, by an ordered set $\\{ \mathbf{q}_i : i \in [0:2^m) \\}$ of $2^m$ distinct binary vectors in  $\mathbb{B}^m$. In this class, a labeling is represented by a matrix $\mathbf{Q} \in \mathbb{B}^{2^m \times m}$, where the $i$-th row of $\mathbf{Q}$ corresponds to the binary vector $\mathbf{q}_i$. For more details, see <cite>SA15, Sec. 2.5.2</cite>.

    Parameters:
        matrix: The labeling matrix $\mathbf{Q}$. Must be a $2^m \times m$ binary matrix whose rows are all distinct.
    """

    def __init__(self, matrix: npt.ArrayLike) -> None:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("'matrix' must be a 2D-array")
        M, m = matrix.shape
        if M != 2**m:
            raise ValueError(f"shape of 'matrix' must be (2**m, m) (got ({M}, {m}))")
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("elements of 'matrix' must be either 0 or 1")
        if len(set(tuple(row) for row in matrix)) != M:
            raise ValueError("rows of 'matrix' must be distinct")
        self._matrix = matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix.tolist()})"

    @property
    def matrix(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.matrix
            array([[1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]])
        """
        return self._matrix

    @property
    def num_bits(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.num_bits
            2
        """
        return super().num_bits

    @property
    def cardinality(self) -> int:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.cardinality
            4
        """
        return super().cardinality

    @property
    def inverse_mapping(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.inverse_mapping
            {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
        """
        return super().inverse_mapping

    def indices_to_bits(self, indices: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.indices_to_bits([2, 0])
            array([0, 1, 1, 0])
            >>> labeling.indices_to_bits([[2, 0], [3, 3]])
            array([[0, 1, 1, 0],
                   [0, 0, 0, 0]])
        """
        return super().indices_to_bits(indices)

    def bits_to_indices(self, bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.bits_to_indices([0, 1, 1, 0])
            array([2, 0])
            >>> labeling.bits_to_indices([[0, 1, 1, 0], [0, 0, 0, 0]])
            array([[2, 0],
                   [3, 3]])
        """
        return super().bits_to_indices(bits)

    def marginalize(self, metrics: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Examples:
            >>> labeling = komm.Labeling([[1, 0], [1, 1], [0, 1], [0, 0]])
            >>> labeling.marginalize([0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25])
            array([0.84729786, 0.        , 0.        , 0.        ])
            >>> labeling.marginalize([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
            array([[0.84729786, 0.        ],
                   [0.        , 0.        ]])
        """
        return super().marginalize(metrics)
