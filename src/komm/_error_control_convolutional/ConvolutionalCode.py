from functools import cache, cached_property
from itertools import product

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import MealyMachine
from .._util.format import format_list_no_quotes as fmt
from .._util.matrices import block_diagonal
from . import base


class ConvolutionalCode(base.ConvolutionalCode):
    r"""
    Binary convolutional encoder. It is characterized by a *matrix of feedforward polynomials* $P(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $q(D)$, of length $k$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively. In this class, the encoder is implemented in controllable canonical form. For more details, see <cite>McE98</cite>, <cite>JZ15</cite>, and <cite>LC04, Chs. 11, 12</cite>.

    Parameters:
        feedforward_polynomials: The matrix of feedforward polynomials $P(D)$, which is a $k \times n$ matrix whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

        feedback_polynomials: The vector of feedback polynomials $q(D)$, which is a $k$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

    Examples:
        1. Consider the encoder with parameters $(n, k, \sigma) = (3, 2, 7)$ depicted below.

            <figure markdown>
            ![Convolutional encoder with parameters (3, 2, 7).](/fig/cc_3_2_7.svg)
            </figure>

            Its matrix of feedforward polynomials is given by
            $$
                P(D) =
                \begin{bmatrix}
                    D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\\\
                    0  &  D^3 + D  &  D^3 + D^2 + 1
                \end{bmatrix}.
            $$

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[
                ...         [0b11001, 0b10111,      0],
                ...         [      0,  0b1010, 0b1101],
                ...     ],
                ... )
                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
                ... )

        1. Consider the feedback encoder with parameters $(n, k, \sigma) = (2, 1, 4)$ depicted below.

            <figure markdown>
            ![Feedback convolutional encoder with parameters (2, 1, 4).](/fig/cc_2_1_4_fb.svg)
            </figure>

            Its matrix of feedforward polynomials is given by
            $$
                P(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1 && D^4 + D^3 + 1
                \end{bmatrix},
            $$
            and its vector of feedback polynomials is given by
            $$
                q(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1
                \end{bmatrix}.
            $$

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0b10111, 0b11001]],
                ...     feedback_polynomials=[0b10111],
                ... )
                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0o27, 0o31]],
                ...     feedback_polynomials=[0o27],
                ... )

    """

    feedforward_polynomials: list[list[BinaryPolynomial]]
    feedback_polynomials: list[BinaryPolynomial]

    def __init__(
        self,
        feedforward_polynomials: npt.ArrayLike,
        feedback_polynomials: npt.ArrayLike | None = None,
    ) -> None:
        ff = np.asarray(feedforward_polynomials, dtype=int)
        if ff.ndim != 2:
            raise ValueError("feedforward must be a 2-dimensional array")
        if feedback_polynomials is None:
            fb = np.ones(ff.shape[0], dtype=int)
        else:
            fb = np.asarray(feedback_polynomials, dtype=int)
        if fb.ndim != 1:
            raise ValueError("feedback must be a 1-dimensional array")
        if fb.shape[0] != ff.shape[0]:
            raise ValueError("feedback and feedforward dimensions do not match")
        self.feedforward_polynomials = [[BinaryPolynomial(p) for p in ps] for ps in ff]
        self.feedback_polynomials = [BinaryPolynomial(q) for q in fb]

    def __repr__(self) -> str:
        args = f"feedforward_polynomials={fmt(self.feedforward_polynomials)}"
        if not np.all([q == 1 for q in self.feedback_polynomials]):
            args += f", feedback_polynomials={fmt(self.feedback_polynomials)}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def num_input_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.num_input_bits
            2
        """
        return len(self.feedforward_polynomials)

    @cached_property
    def num_output_bits(self) -> int:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.num_output_bits
            3
        """
        return len(self.feedforward_polynomials[0])

    @cached_property
    def degree(self) -> int:
        r"""
        It is given by
        $$
            \sigma = \sum_{i \in [0:k)} \nu_i,
        $$
        where $\nu_i$ are the constraint lengths of the encoder.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.degree
            7
        """
        return int(np.sum(self.constraint_lengths))

    @cached_property
    def memory_order(self) -> int:
        r"""
        The *memory order* $\mu$ of the encoder. It is given by
        $$
            \mu = \max_{i \in [0:k)} \nu_i,
        $$
        where $\nu_i$ are the constraint lengths of the encoder.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.memory_order
            4
        """
        return int(np.max(self.constraint_lengths))

    @cached_property
    def constraint_lengths(self) -> npt.NDArray[np.integer]:
        r"""
        The *constraint lengths* $\nu_i$ of the encoder, defined by
        $$
            \nu_i = \max \\{ \deg p_{i,0}(D), \deg p_{i,1}(D), \ldots, \deg p_{i,n-1}(D), \deg q_i(D) \\},
        $$
        for $i \in [0 : k)$. This is a $k$-array of integers.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.constraint_lengths
            array([4, 3])
        """
        nus = np.empty(self.num_input_bits, dtype=int)
        for i in range(self.num_input_bits):
            ps = self.feedforward_polynomials[i]
            q = self.feedback_polynomials[i]
            nus[i] = max(np.amax([p.degree for p in ps]), q.degree)
        return nus

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
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
            >>> A_mat
            array([[0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0]])
            >>> B_mat
            array([[1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0]])
            >>> C_mat
            array([[0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 1, 1]])
            >>> D_mat
            array([[1, 1, 0],
                   [0, 0, 1]])

            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o27, 0o31]],
            ...     feedback_polynomials=[0o27],
            ... )
            >>> A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
            >>> A_mat
            array([[1, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]])
            >>> B_mat
            array([[1, 0, 0, 0]])
            >>> C_mat
            array([[0, 1],
                   [0, 1],
                   [0, 1],
                   [0, 0]])
            >>> D_mat
            array([[1, 1]])
        """
        betas = [
            np.array([p.coefficients(nu + 1) for p in ps]).T
            for nu, ps in zip(self.constraint_lengths, self.feedforward_polynomials)
        ]

        alphas = [
            q.coefficients(nu + 1)[:, np.newaxis]
            for nu, q in zip(self.constraint_lengths, self.feedback_polynomials)
        ]

        A_blocks: list[npt.NDArray[np.integer]] = []
        B_blocks: list[npt.NDArray[np.integer]] = []
        C_blocks: list[npt.NDArray[np.integer]] = []
        D_blocks: list[npt.NDArray[np.integer]] = []

        for alpha, beta, nu in zip(alphas, betas, self.constraint_lengths):
            if nu == 0:
                A_blocks.append(np.zeros((0, 0), dtype=int))
            else:
                A_blocks.append(np.hstack([alpha[1:], np.eye(nu, nu - 1, dtype=int)]))
            B_blocks.append(np.eye(1, nu, dtype=int))
            C_blocks.append(beta[1:] ^ (alpha[1:] * beta[0]))
            D_blocks.append(beta[0])

        A_mat = block_diagonal(A_blocks)
        B_mat = block_diagonal(B_blocks)
        C_mat = np.vstack(C_blocks)
        D_mat = np.vstack(D_blocks)

        return A_mat, B_mat, C_mat, D_mat

    @cache
    def generator_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        For a convolutional code with matrix of feedforward polynomials
        $$
            P(D) =
            \begin{bmatrix}
                p_{0,0}(D)   & p_{0,1}(D)   & \cdots & p_{0,n-1}(D)   \\\\
                p_{1,0}(D)   & p_{1,1}(D)   & \cdots & p_{1,n-1}(D)   \\\\
                \vdots       & \vdots       & \ddots & \vdots         \\\\
                p_{k-1,0}(D) & p_{k-1,1}(D) & \cdots & p_{k-1,n-1}(D)
            \end{bmatrix},
        $$
        and vector of feedback polynomials
        $$
            q(D) =
            \begin{bmatrix}
                q_0(D)     \\\\
                q_1(D)     \\\\
                \vdots     \\\\
                q_{k-1}(D)
            \end{bmatrix},
        $$
        the generator matrix is given by
        $$
            G(D) =
            \begin{bmatrix}
                p_{0,0}(D)/q_0(D)       & p_{0,1}(D)/q_0(D)       & \cdots & p_{0,n-1}(D)/q_0(D)       \\\\
                p_{1,0}(D)/q_1(D)       & p_{1,1}(D)/q_1(D)       & \cdots & p_{1,n-1}(D)/q_1(D)       \\\\
                \vdots                  & \vdots                  & \ddots & \vdots                    \\\\
                p_{k-1,0}(D)/q_{k-1}(D) & p_{k-1,1}(D)/q_{k-1}(D) & \cdots & p_{k-1,n-1}(D)/q_{k-1}(D)
            \end{bmatrix}.
        $$

        Examples:
            If matrix of feedforward polynomials is
            $$
                P(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1 && D^4 + D^3 + 1
                \end{bmatrix}
            $$
            and vector of feedback polynomials is
            $$
                q(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1
                \end{bmatrix},
            $$
            then the generator matrix is given by
            $$
                G(D) =
                \begin{bmatrix}
                    1 & \frac{D^4 + D^3 + 1}{D^4 + D^2 + D + 1}
                \end{bmatrix}.
            $$

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0o27, 0o31]],
                ...     feedback_polynomials=[0o27],
                ... )
                >>> for row in code.generator_matrix():
                ...     print("[" + ", ".join(str(x) for x in row) + "]")
                [0b1/0b1, 0b11001/0b10111]
        """
        n, k = self.num_output_bits, self.num_input_bits
        G_mat = np.empty((k, n), dtype=object)
        for i, j in product(range(k), range(n)):
            p = BinaryPolynomialFraction(self.feedforward_polynomials[i][j])
            q = BinaryPolynomialFraction(self.feedback_polynomials[i])
            G_mat[i, j] = p / q
        return G_mat

    @cache
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.finite_state_machine()
            MealyMachine(transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
                         outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
        """
        return super().finite_state_machine()

    @cache
    def is_catastrophic(self) -> bool:
        """
        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.is_catastrophic()
            False

            >>> code = komm.ConvolutionalCode([[0b11, 0b101]])
            >>> code.is_catastrophic()
            True
        """
        return super().is_catastrophic()

    @cache
    def free_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
            ... )
            >>> code.free_distance()
            5
        """
        return super().free_distance()

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.encode([1, 1, 1, 1])
            array([1, 1, 0, 1, 1, 0, 1, 0])
        """
        return super().encode(input)

    def encode_with_state(
        self,
        input: npt.ArrayLike,
        initial_state: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        r"""
        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.encode_with_state([1, 1, 1, 1], [0, 0])
            (array([1, 1, 0, 1, 1, 0, 1, 0]), array([1, 1]))
            >>> code.encode_with_state([1, 1, 1, 1], [1, 1])
            (array([1, 0, 1, 0, 1, 0, 1, 0]), array([1, 1]))
        """
        return super().encode_with_state(input, initial_state)
