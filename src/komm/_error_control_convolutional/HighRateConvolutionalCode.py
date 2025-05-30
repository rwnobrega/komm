from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import MealyMachine
from .._util.format import format_list_no_quotes as fmt
from . import base


class HighRateConvolutionalCode(base.ConvolutionalCode):
    r"""
    High-rate convolutional encoder. It is an $(n, n-1)$ recursive systematic [convolutional encoder](/ref/ConvolutionalCode) defined by a single *check row* $h(D) \in \mathbb{F}_2[D]^n$ and realized in observable canonical form. By convention, the first $n - 1$ positions represent the information bits.

    Parameters:
        h_row: The check row $h(D)$ of the encoder. Must be an $n$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

    Examples:
        Consider the high-rate convolutional encoder with $(n, k, \sigma) = (4, 3, 3)$ depicted below.

        <figure markdown>/
        ![Convolutional encoder for high-rate (4, 3, 3) code.](/fig/cc_high_rate_4_3_3.svg)
        </figure>

        Its check row is given by
        $$
            h(D) =
            \begin{bmatrix}
                D^3 + D  &&  D^3 + D^2 + 1  &&  D^3 + D + 1  &&  D^3 + 1
            \end{bmatrix}.
        $$

            >>> code = komm.HighRateConvolutionalCode([0b1010, 0b1101, 0b1011, 0b1001])
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])

    Please refer to [the table of optimal high-rate convolutional codes](/res/convolutional-codes/#high-rate).
    """

    h_row: list[BinaryPolynomial]

    def __init__(self, h_row: npt.ArrayLike) -> None:
        h_row = np.asarray(h_row, dtype=int)
        if h_row.ndim != 1:
            raise ValueError("'h_row' must be a 1-dimensional array")
        self.h_row = [BinaryPolynomial(x) for x in h_row]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({fmt(self.h_row)})"

    @cached_property
    def num_input_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.num_input_bits
            3
        """
        return len(self.h_row) - 1

    @cached_property
    def num_output_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.num_output_bits
            4
        """
        return len(self.h_row)

    @cached_property
    def degree(self) -> int:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.degree
            3
        """
        return max(p.degree for p in self.h_row)

    @cache
    def state_space_representation(
        self,
    ) -> tuple[
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
    ]:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
            >>> A_mat
            array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
            >>> B_mat
            array([[1, 0, 1],
                   [0, 1, 0],
                   [0, 0, 1]])
            >>> C_mat
            array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1]])
            >>> D_mat
            array([[1, 0, 0, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 1]])
        """
        n, σ = self.num_output_bits, self.degree
        beta = np.array([p.coefficients(σ + 1) for p in self.h_row[:-1]])
        beta = np.fliplr(beta)
        alpha = np.array([self.h_row[-1].coefficients(σ + 1)])
        alpha = np.fliplr(alpha)
        A_mat = np.vstack([np.eye(σ - 1, σ, k=1, dtype=int), alpha[:, :-1]])
        B_mat = np.empty((n - 1, σ), dtype=int)
        for j in range(σ):
            B_mat[:, j] = beta[:, j] ^ (alpha[0, j] * beta[:, -1])
        C_mat = np.zeros((σ, n), dtype=int)
        C_mat[-1, -1] = 1
        D_mat = np.hstack([np.eye(n - 1, dtype=int), beta[:, -1:]])
        return A_mat, B_mat, C_mat, D_mat

    @cache
    def generator_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        For a high-rate convolutional code with check row
        $$
            h(D) =
            \begin{bmatrix}
                h_0(D)  &&  h_1(D)  &&  \cdots  &&  h_{n-1}(D)
            \end{bmatrix},
        $$
        the generator matrix is given by
        $$
            G(D) =
            \begin{bmatrix}
                1      & 0      & \cdots & 0      & h_0(D) / h_{n-1}(D)     \\\\[1ex]
                0      & 1      & \cdots & 0      & h_1(D) / h_{n-1}(D)     \\\\[1ex]
                \vdots & \vdots & \ddots & \vdots & \vdots                  \\\\[1ex]
                0      & 0      & \cdots & 1      & h_{n-2}(D) / h_{n-1}(D)
            \end{bmatrix}.
        $$

        Examples:
            If the check row is
            $$
                h(D) =
                \begin{bmatrix}
                    D^3 + D  &&  D^3 + D^2 + 1  &&  D^3 + D + 1  &&  D^3 + 1
                \end{bmatrix},
            $$
            then the generator matrix is
            $$
                G(D) =
                \begin{bmatrix}
                    1 & 0 & 0 & \frac{D^3 + D}{D^3 + 1} \\\\[1ex]
                    0 & 1 & 0 & \frac{D^3 + D^2 + 1}{D^3 + 1} \\\\[1ex]
                    0 & 0 & 1 & \frac{D^3 + D + 1}{D^3 + 1}
                \end{bmatrix}.
            $$

                >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
                >>> for row in code.generator_matrix():
                ...     print("[" + ", ".join(str(x).ljust(12) for x in row) + "]")
                [0b1/0b1     , 0b0/0b1     , 0b0/0b1     , 0b110/0b111 ]
                [0b0/0b1     , 0b1/0b1     , 0b0/0b1     , 0b1101/0b1001]
                [0b0/0b1     , 0b0/0b1     , 0b1/0b1     , 0b1011/0b1001]
        """
        n, k = self.num_output_bits, self.num_input_bits
        G_mat = np.empty((k, n), dtype=object)
        for i in range(n - 1):
            for j in range(n - 1):
                G_mat[i, j] = BinaryPolynomialFraction(int(i == j))
            G_mat[i, -1] = BinaryPolynomialFraction(self.h_row[i], self.h_row[-1])
        return G_mat

    @cache
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> fsm = code.finite_state_machine()
            >>> (fsm.num_input_symbols, fsm.num_output_symbols, fsm.num_states)
            (8, 16, 8)
        """
        return super().finite_state_machine()

    @cache
    def is_catastrophic(self) -> bool:
        """
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.is_catastrophic()
            False
        """
        return super().is_catastrophic()

    @cache
    def free_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.free_distance()
            4
        """
        return super().free_distance()

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.encode([1, 1, 1, 1, 1, 1])
            array([1, 1, 1, 0, 1, 1, 1, 0])
        """
        return super().encode(input)

    def encode_with_state(
        self,
        input: npt.ArrayLike,
        initial_state: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        r"""
        Examples:
            >>> code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
            >>> code.encode_with_state([1, 1, 1, 1, 1, 1], [0, 0, 0])
            (array([1, 1, 1, 0, 1, 1, 1, 0]), array([1, 0, 1]))
            >>> code.encode_with_state([1, 1, 1, 1, 1, 1], [1, 0, 1])
            (array([1, 1, 1, 1, 1, 1, 1, 0]), array([1, 1, 0]))
        """
        return super().encode_with_state(input, initial_state)
