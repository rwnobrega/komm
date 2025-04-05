from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import FiniteStateMachine
from .._util.array import array
from .._util.bit_operations import bits_to_int, int_to_bits


class ConvolutionalCode:
    r"""
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* $\mathbf{P}(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $\mathbf{q}(D)$, of length $k$. The element in row $i$ and column $j$ of $\mathbf{P}(D)$ is denoted by $p_{i,j}(D)$, and the element in position $i$ of $\mathbf{q}(D)$ is denoted by $q_i(D)$; they are [binary polynomials](/ref/BinaryPolynomial) in $D$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively.

    The *transfer function matrix* (also known as *transform-domain generator matrix*) $\mathbf{G}(D)$ of the convolutional code, of shape $k \times n$, is such that the element in row $i$ and column $j$ is given by
    $$
        g_{i,j}(D) = \frac{p_{i,j}(D)}{q_{i}(D)},
    $$
    for $i \in [0 : k)$ and $j \in [0 : n)$.

    The *constraint lengths* of the code are defined by
    $$
        \nu_i = \max \\{ \deg p_{i,0}(D), \deg p_{i,1}(D), \ldots, \deg p_{i,n-1}(D), \deg q_i(D) \\},
    $$
    for $i \in [0 : k)$.

    The *overall constraint length* of the code is defined by
    $$
        \nu = \sum_{0 \leq i < k} \nu_i.
    $$

    The *memory order* of the code is defined by
    $$
        \mu = \max_{0 \leq i < k} \nu_i.
    $$

    For more details, see <cite>JZ15</cite> and <cite>LC04, Chs. 11, 12</cite>.

    Parameters:
        feedforward_polynomials: The matrix of feedforward polynomials $\mathbf{P}(D)$, which is a $k \times n$ matrix whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

        feedback_polynomials: The vector of feedback polynomials $\mathbf{q}(D)$, which is a $k$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

    Examples:
        1. The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 6)$; its transfer function matrix is given by
            $$
                \mathbf{G}(D) =
                \begin{bmatrix}
                    D^6 + D^3 + D^2 + D + 1  &  D^6 + D^5 + D^3 + D^2 + 1
                \end{bmatrix},
            $$
            yielding `feedforward_polynomials = [[0b1001111, 0b1101101]] = [[0o117, 0o155]] = [[79, 109]]`.

            <figure markdown>
            ![Convolutional encoder for (2, 1, 6) code.](/figures/cc_2_1_6.svg)
            </figure>

                >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o117, 0o155]])
                >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
                (2, 1, 6)

        1. The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (3, 2, 7)$; its transfer function matrix is given by
            $$
                \mathbf{G}(D) =
                \begin{bmatrix}
                    D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\\\
                    0  &  D^3 + D  &  D^3 + D^2 + 1
                \end{bmatrix},
            $$
            yielding `feedforward_polynomials = [[0b11001, 0b10111, 0b00000], [0b0000, 0b1010, 0b1101]] = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]] = [[25, 23, 0], [0, 10, 13]]`.

            <figure markdown>
            ![Convolutional encoder for (3, 2, 7) code.](/figures/cc_3_2_7.svg)
            </figure>

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
                ... )
                >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
                (3, 2, 7)

        1. The convolutional code with feedback encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 4)$; its transfer function matrix is given by
            $$
                \mathbf{G}(D) =
                \begin{bmatrix}
                    1  &  \dfrac{D^4 + D^3 + 1}{D^4 + D^2 + D + 1}
                \end{bmatrix},
            $$
            yielding `feedforward_polynomials = [[0b10111, 0b11001]] = [[0o27, 0o31]] = [[23, 25]]` and `feedback_polynomials = [0b10111] = [0o27] = [23]`.

            <figure markdown>
            ![Convolutional encoder for (2, 1, 4) feedback code.](/figures/cc_2_1_4_fb.svg)
            </figure>

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0o27, 0o31]],
                ...     feedback_polynomials=[0o27],
                ... )
                >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
                (2, 1, 4)

    <h2>Tables of optimal convolutional codes</h2>

    The tables below <cite>LC04, Sec. 12.3</cite> lists optimal convolutional codes with no feedback, for parameters $(n,k) = (2,1)$ and $(n,k) = (3,1)$, and small values of the overall constraint length $\nu$.

    | Parameters $(n, k, \nu)$ | Transfer function matrix $\mathbf{G}(D)$ |
    | :----------------------: | ---------------------------------------- |
    | $(2, 1, 1)$              | `[[0o1, 0o3]]`                           |
    | $(2, 1, 2)$              | `[[0o5, 0o7]]`                           |
    | $(2, 1, 3)$              | `[[0o13, 0o17]]`                         |
    | $(2, 1, 4)$              | `[[0o27, 0o31]]`                         |
    | $(2, 1, 5)$              | `[[0o53, 0o75]]`                         |
    | $(2, 1, 6)$              | `[[0o117, 0o155]]`                       |
    | $(2, 1, 7)$              | `[[0o247, 0o371]]`                       |
    | $(2, 1, 8)$              | `[[0o561, 0o753]]`                       |

    | Parameters $(n, k, \nu)$ | Transfer function matrix $\mathbf{G}(D)$ |
    | :----------------------: | ---------------------------------------- |
    | $(3, 1, 1)$              | `[[0o1, 0o3, 0o3]]`                      |
    | $(3, 1, 2)$              | `[[0o5, 0o7, 0o7]]`                      |
    | $(3, 1, 3)$              | `[[0o13, 0o15, 0o17]]`                   |
    | $(3, 1, 4)$              | `[[0o25, 0o33, 0o37]]`                   |
    | $(3, 1, 5)$              | `[[0o47, 0o53, 0o75]]`                   |
    | $(3, 1, 6)$              | `[[0o117, 0o127, 0o155]]`                |
    | $(3, 1, 7)$              | `[[0o255, 0o331, 0o367]]`                |
    | $(3, 1, 8)$              | `[[0o575, 0o623, 0o727]]`                |
    """

    feedforward_polynomials: npt.NDArray[np.object_]
    feedback_polynomials: npt.NDArray[np.object_]

    def __init__(
        self,
        feedforward_polynomials: npt.ArrayLike,
        feedback_polynomials: npt.ArrayLike | None = None,
    ) -> None:
        self.feedforward_polynomials = array(feedforward_polynomials, BinaryPolynomial)
        if self.feedforward_polynomials.ndim != 2:
            raise ValueError("feedforward must be a 2-dimensional array")
        k = self.feedforward_polynomials.shape[0]
        if feedback_polynomials is None:
            self.feedback_polynomials = array([1] * k, BinaryPolynomial)
        else:
            self.feedback_polynomials = array(feedback_polynomials, BinaryPolynomial)
        if self.feedback_polynomials.ndim != 1:
            raise ValueError("feedback must be a 1-dimensional array")
        if self.feedback_polynomials.shape[0] != k:
            raise ValueError("feedback and feedforward dimensions do not match")

    def __repr__(self) -> str:
        args = f"feedforward_polynomials={self.feedforward_polynomials}"
        if not np.all(self.feedback_polynomials == 1):
            args += f", feedback_polynomials={self.feedback_polynomials}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def num_input_bits(self) -> int:
        r"""
        The number of input bits per block, $k$.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.num_input_bits
            2
        """
        return self.feedforward_polynomials.shape[0]

    @cached_property
    def num_output_bits(self) -> int:
        r"""
        The number of output bits per block, $n$.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.num_output_bits
            3
        """
        return self.feedforward_polynomials.shape[1]

    @cached_property
    def transfer_function_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        The transfer function matrix $\mathbf{G}(D)$ of the code. This is a $k \times n$ array of [binary polynomial fractions](/ref/BinaryPolynomialFraction).

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> for row in code.transfer_function_matrix:
            ...     print("[" + ", ".join(str(x).ljust(12) for x in row) + "]")
            [0b11001/0b1 , 0b10111/0b1 , 0b0/0b1     ]
            [0b0/0b1     , 0b1010/0b1  , 0b1101/0b1  ]

            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o27, 0o31]],
            ...     feedback_polynomials=[0o27],
            ... )
            >>> for row in code.transfer_function_matrix:
            ...     print("[" + ", ".join(str(x) for x in row) + "]")
            [0b1/0b1, 0b11001/0b10111]
        """
        transfer_function_matrix = np.empty_like(self.feedforward_polynomials)
        for i, j in np.ndindex(self.feedforward_polynomials.shape):
            p = BinaryPolynomialFraction(self.feedforward_polynomials[i, j])
            q = BinaryPolynomialFraction(self.feedback_polynomials[i])
            transfer_function_matrix[i, j] = p / q
        return transfer_function_matrix

    @cached_property
    def constraint_lengths(self) -> npt.NDArray[np.integer]:
        r"""
        The constraint lengths $\nu_i$ of the code, for $i \in [0 : k)$. This is a $k$-array of integers.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
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

    @cached_property
    def overall_constraint_length(self) -> int:
        r"""
        The overall constraint length $\nu$ of the code.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.overall_constraint_length
            7
        """
        return int(np.sum(self.constraint_lengths))

    @cached_property
    def memory_order(self) -> int:
        r"""
        The memory order $\mu$ of the code.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.memory_order
            4
        """
        return int(np.max(self.constraint_lengths))

    @cache
    def finite_state_machine(self) -> FiniteStateMachine:
        r"""
        Returns the [finite-state machine](/ref/FiniteStateMachine) of the code, in direct form.

        Returns:
            The finite-state machine of the code.

        Examples:
            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0b101, 0b111]])
            >>> code.finite_state_machine()
            FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]],
                               outputs=[[0, 3], [2, 1], [3, 0], [1, 2]])
        """
        n = self.num_output_bits
        k = self.num_input_bits
        nu = self.overall_constraint_length

        x_indices = np.concatenate(([0], np.cumsum(self.constraint_lengths + 1)[:-1]))
        s_indices = np.setdiff1d(np.arange(k + nu, dtype=int), x_indices)

        ff_taps: list[npt.NDArray[np.integer]] = []
        for j in range(n):
            taps = np.concatenate([
                self.feedforward_polynomials[i, j].exponents() + x_indices[i]
                for i in range(k)
            ])
            ff_taps.append(taps)

        fb_taps: list[npt.NDArray[np.integer]] = []
        for i in range(k):
            taps = (
                BinaryPolynomial(0b1) + self.feedback_polynomials[i]
            ).exponents() + x_indices[i]
            fb_taps.append(taps)

        bits = np.empty(k + nu, dtype=int)
        next_states = np.empty((2**nu, 2**k), dtype=int)
        outputs = np.empty((2**nu, 2**k), dtype=int)

        for s, x in np.ndindex(2**nu, 2**k):
            bits[s_indices] = int_to_bits(s, width=nu)
            bits[x_indices] = int_to_bits(x, width=k)
            bits[x_indices] ^= [np.sum(bits[fb_taps[i]]) % 2 for i in range(k)]

            next_state_bits = bits[s_indices - 1]
            output_bits = [np.sum(bits[ff_taps[j]]) % 2 for j in range(n)]

            next_states[s, x] = bits_to_int(next_state_bits)
            outputs[s, x] = bits_to_int(output_bits)

        return FiniteStateMachine(next_states, outputs)

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
        Returns the *state-space representation* of the code. Let
        $$
        \begin{aligned}
            \mathbf{u}_t & = (u_t^{(0)}, u_t^{(1)}, \ldots, u_t^{(k-1)}), \\\\
            \mathbf{v}_t & = (v_t^{(0)}, v_t^{(1)}, \ldots, v_t^{(n-1)}), \\\\
            \mathbf{s}_t & = (s_t^{(0)}, s_t^{(1)}, \ldots, s_t^{(\nu-1)}),
        \end{aligned}
        $$
        be the input block, output block, and state, respectively, all defined at time instant $t$. Then,
        $$
        \begin{aligned}
            \mathbf{s}\_{t+1} & = \mathbf{s}_t \mathbf{A} + \mathbf{u}_t \mathbf{B}, \\\\
            \mathbf{v}\_{t} & = \mathbf{s}_t \mathbf{C} + \mathbf{u}\_t \mathbf{D},
        \end{aligned}
        $$
        where $\mathbf{A}$ is the $\nu \times \nu$ *state matrix*, $\mathbf{B}$ is the $k \times \nu$ *control matrix*, $\mathbf{C}$ is the $\nu \times n$ *observation matrix*, and $\mathbf{D}$ is the $k \times n$ *transition matrix*. They are all binary matrices. For more details, see <cite>WBR01</cite>.

        Returns:
            The state matrix $\mathbf{A}$ of the code.
            The control matrix $\mathbf{B}$ of the code.
            The observation matrix $\mathbf{C}$ of the code.
            The transition matrix $\mathbf{D}$ of the code.

        Examples:
            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0b101, 0b111]])
            >>> state_matrix, control_matrix, observation_matrix, transition_matrix = (
            ...     code.state_space_representation()
            ... )
            >>> state_matrix
            array([[0, 1],
                   [0, 0]])
            >>> control_matrix
            array([[1, 0]])
            >>> observation_matrix
            array([[0, 1],
                   [1, 1]])
            >>> transition_matrix
            array([[1, 1]])
        """
        k = self.num_input_bits
        n = self.num_output_bits
        nu = self.overall_constraint_length
        fsm = self.finite_state_machine()

        state_matrix = np.empty((nu, nu), dtype=int)
        observation_matrix = np.empty((nu, n), dtype=int)
        for i in range(nu):
            s0 = 2**i
            state_matrix[i, :] = int_to_bits(fsm.next_states[s0, 0], width=nu)
            observation_matrix[i, :] = int_to_bits(fsm.outputs[s0, 0], width=n)

        control_matrix = np.empty((k, nu), dtype=int)
        transition_matrix = np.empty((k, n), dtype=int)
        for i in range(k):
            x = 2**i
            control_matrix[i, :] = int_to_bits(fsm.next_states[0, x], width=nu)
            transition_matrix[i, :] = int_to_bits(fsm.outputs[0, x], width=n)

        return state_matrix, control_matrix, observation_matrix, transition_matrix
