from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import MealyMachine
from .._util.format import format_list_no_quotes as fmt
from . import base


class LowRateConvolutionalCode(base.ConvolutionalCode):
    r"""
    Low-rate convolutional encoder. It is an $(n, 1)$ non-recursive non-systematic [convolutional encoder](/ref/ConvolutionalCode) defined by a single *generator row* $g(D) \in \mathbb{F}_2[D]^n$ and realized in controllable canonical form.

    Parameters:
        g_row: The generator row $g(D)$ of the encoder. Must be an $n$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

    Examples:
        Consider the low-rate convolutional encoder with $(n, k, \sigma) = (2, 1, 6)$ depicted below.

        <figure markdown>
        ![Convolutional encoder for low-rate (2, 1, 6) code.](/fig/cc_low_rate_2_1_6.svg)
        </figure>

        Its generator row is given by
        $$
            g(D) =
            \begin{bmatrix}
                D^6 + D^3 + D^2 + D + 1  &&  D^6 + D^5 + D^3 + D^2 + 1
            \end{bmatrix}.
        $$

            >>> code = komm.LowRateConvolutionalCode([0b1001111, 0b1101101])
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])

    Please refer to [the table of optimal low-rate convolutional codes](/res/convolutional-codes/#low-rate).
    """

    g_row: list[BinaryPolynomial]

    def __init__(self, g_row: npt.ArrayLike) -> None:
        g_row = np.asarray(g_row, dtype=int)
        if g_row.ndim != 1:
            raise ValueError("'g_row' must be a 1-dimensional array")
        self.g_row = [BinaryPolynomial(x) for x in g_row]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({fmt(self.g_row)})"

    @cached_property
    def num_input_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.num_input_bits
            1
        """
        return 1

    @cached_property
    def num_output_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.num_output_bits
            2
        """
        return len(self.g_row)

    @cached_property
    def degree(self) -> int:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.degree
            6
        """
        return max(p.degree for p in self.g_row)

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
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
            >>> A_mat
            array([[0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0]])
            >>> B_mat
            array([[1, 0, 0, 0, 0, 0]])
            >>> C_mat
            array([[1, 0],
                   [1, 1],
                   [1, 1],
                   [0, 0],
                   [0, 1],
                   [1, 1]])
            >>> D_mat
            array([[1, 1]])
        """
        σ = self.degree
        beta = np.array([p.coefficients(σ + 1) for p in self.g_row]).T
        A_mat = np.eye(σ, σ, k=1, dtype=int)
        B_mat = np.eye(1, σ, dtype=int)
        C_mat = beta[1:]
        D_mat = np.array([beta[0]])
        return A_mat, B_mat, C_mat, D_mat

    @cache
    def generator_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        For a low-rate convolutional code, it is given by
        $$
            G(D) = \big[ ~ g(D) ~ \big].
        $$

        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> for row in code.generator_matrix():
            ...     print("[" + ", ".join(str(x) for x in row) + "]")
            [0b1001111/0b1, 0b1101101/0b1]
        """
        n, k = self.num_output_bits, self.num_input_bits
        G_mat = np.empty((k, n), dtype=object)
        for j in range(n):
            p = BinaryPolynomialFraction(self.g_row[j])
            G_mat[0, j] = p
        return G_mat

    @cache
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> fsm = code.finite_state_machine()
            >>> (fsm.num_input_symbols, fsm.num_output_symbols, fsm.num_states)
            (2, 4, 64)
        """
        return super().finite_state_machine()

    @cache
    def is_catastrophic(self) -> bool:
        """
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.is_catastrophic()
            False
        """
        return super().is_catastrophic()

    @cache
    def free_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.free_distance()
            10
        """
        return super().free_distance()

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.encode([1, 1, 1, 1])
            array([1, 1, 0, 1, 1, 0, 0, 1])
        """
        return super().encode(input)

    def encode_with_state(
        self,
        input: npt.ArrayLike,
        initial_state: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        r"""
        Examples:
            >>> code = komm.LowRateConvolutionalCode([0o117, 0o155])
            >>> code.encode_with_state([1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
            (array([1, 1, 0, 1, 1, 0, 0, 1]), array([1, 1, 1, 1, 0, 0]))
            >>> code.encode_with_state([1, 1, 1, 1], [1, 1, 1, 1, 0, 0])
            (array([0, 1, 0, 0, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1]))
        """
        return super().encode_with_state(input, initial_state)
