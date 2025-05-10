from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import FiniteStateMachine
from .._util.array import array
from .._util.bit_operations import bits_to_int, int_to_bits, xor


class ConvolutionalCode:
    r"""
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* $\mathbf{P}(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $\mathbf{q}(D)$, of length $k$. The element in row $i$ and column $j$ of $\mathbf{P}(D)$ is denoted by $p_{i,j}(D)$, and the element in position $i$ of $\mathbf{q}(D)$ is denoted by $q_i(D)$; they are [binary polynomials](/ref/BinaryPolynomial) in $D$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively. For more details, see <cite>JZ15</cite> and <cite>LC04, Chs. 11, 12</cite>.

    Parameters:
        feedforward_polynomials: The matrix of feedforward polynomials $\mathbf{P}(D)$, which is a $k \times n$ matrix whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

        feedback_polynomials: The vector of feedback polynomials $\mathbf{q}(D)$, which is a $k$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

    Examples:
        1. Consider the encoder with parameters $(n, k, \nu) = (2, 1, 6)$ depicted below.

            <figure markdown>
            ![Convolutional encoder for (2, 1, 6) code.](/figures/cc_2_1_6.svg)
            </figure>

            Its matrix of feedforward polynomials is given by
            $$
                \mathbf{P}(D) =
                \begin{bmatrix}
                    D^6 + D^3 + D^2 + D + 1  &&  D^6 + D^5 + D^3 + D^2 + 1
                \end{bmatrix}.
            $$

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0b1001111, 0b1101101]],
                ... )

        1. Consider the encoder with parameters $(n, k, \nu) = (3, 2, 7)$ depicted below.

            <figure markdown>
            ![Convolutional encoder for (3, 2, 7) code.](/figures/cc_3_2_7.svg)
            </figure>

            Its matrix of feedforward polynomials is given by
            $$
                \mathbf{P}(D) =
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

        1. Consider the feedback encoder with parameters $(n, k, \nu) = (2, 1, 4)$ depicted below.

            <figure markdown>
            ![Convolutional encoder for (2, 1, 4) feedback code.](/figures/cc_2_1_4_fb.svg)
            </figure>

            Its matrix of feedforward polynomials is given by
            $$
                \mathbf{P}(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1 && D^4 + D^3 + 1
                \end{bmatrix},
            $$
            and its vector of feedback polynomials is given by
            $$
                \mathbf{q}(D) =
                \begin{bmatrix}
                    D^4 + D^2 + D + 1
                \end{bmatrix}.
            $$

                >>> code = komm.ConvolutionalCode(
                ...     feedforward_polynomials=[[0b10111, 0b11001]],
                ...     feedback_polynomials=[0b10111],
                ... )

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
        self.__post_init__()

    def __post_init__(self) -> None:
        n = self.num_output_bits
        k = self.num_input_bits
        nu = self.overall_constraint_length

        u_indices = np.concatenate(([0], np.cumsum(self.constraint_lengths + 1)[:-1]))
        self._u_indices = u_indices
        s_indices = np.setdiff1d(np.arange(k + nu, dtype=int), u_indices)
        self._s_indices = s_indices

        self._ff_taps: list[npt.NDArray[np.integer]] = []
        for j in range(n):
            taps = np.concatenate([
                self.feedforward_polynomials[i, j].exponents() + u_indices[i]
                for i in range(k)
            ])
            self._ff_taps.append(taps)

        self._fb_taps: list[npt.NDArray[np.integer]] = []
        for i in range(k):
            taps = (
                BinaryPolynomial(0b1) + self.feedback_polynomials[i]
            ).exponents() + u_indices[i]
            self._fb_taps.append(taps)

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
        The *transfer function matrix* (also known as *transform-domain generator matrix*) $\mathbf{G}(D)$ of the code. This is a $k \times n$ array of [binary polynomial fractions](/ref/BinaryPolynomialFraction) with element in row $i$ and column $j$ given by
        $$
            g_{i,j}(D) = \frac{p_{i,j}(D)}{q_{i}(D)},
        $$
        for $i \in [0 : k)$ and $j \in [0 : n)$.

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
        The *constraint lengths* $\nu_i$ of the code, defined by
        $$
            \nu_i = \max \\{ \deg p_{i,0}(D), \deg p_{i,1}(D), \ldots, \deg p_{i,n-1}(D), \deg q_i(D) \\},
        $$
        for $i \in [0 : k)$. This is a $k$-array of integers.

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
        The *overall constraint length* of the code, defined by
        $$
            \nu = \sum_{i \in [0:k)} \nu_i.
        $$

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
        The *memory order* of the code, defined by
        $$
            \mu = \max_{i \in [0:k)} \nu_i.
        $$

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.memory_order
            4
        """
        return int(np.max(self.constraint_lengths))

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
        Returns the *state-space representation* of the code, in controller canonical form. Let
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
            state_matrix: The state matrix $\mathbf{A}$ of the code.
            control_matrix: The control matrix $\mathbf{B}$ of the code.
            observation_matrix: The observation matrix $\mathbf{C}$ of the code.
            transition_matrix: The transition matrix $\mathbf{D}$ of the code.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> state_matrix, control_matrix, observation_matrix, transition_matrix = (
            ...     code.state_space_representation()
            ... )
            >>> state_matrix
            array([[0, 1],
                   [0, 0]])
            >>> control_matrix
            array([[1, 0]])
            >>> observation_matrix
            array([[1, 0],
                   [1, 1]])
            >>> transition_matrix
            array([[1, 1]])
        """
        n, k = self.num_output_bits, self.num_input_bits
        nu = self.overall_constraint_length
        u_indices, s_indices = self._u_indices, self._s_indices
        ff_taps, fb_taps = self._ff_taps, self._fb_taps

        bits = np.empty(k + nu, dtype=int)
        A_mat = np.zeros((nu, nu), dtype=int)
        B_mat = np.zeros((k, nu), dtype=int)
        C_mat = np.zeros((nu, n), dtype=int)
        D_mat = np.zeros((k, n), dtype=int)

        for row, s in enumerate(np.eye(nu)):
            bits[s_indices] = s
            bits[u_indices] = 0
            bits[u_indices] ^= [xor(bits[fb_taps[i]]) for i in range(k)]
            A_mat[row] = bits[s_indices - 1]
            C_mat[row] = [xor(bits[ff_taps[j]]) for j in range(n)]

        for row, u in enumerate(np.eye(k)):
            bits[s_indices] = 0
            bits[u_indices] = u
            bits[u_indices] ^= [xor(bits[fb_taps[i]]) for i in range(k)]
            B_mat[row] = bits[s_indices - 1]
            D_mat[row] = [xor(bits[ff_taps[j]]) for j in range(n)]

        return A_mat, B_mat, C_mat, D_mat

    @cache
    def finite_state_machine(self) -> FiniteStateMachine:
        r"""
        Returns the [finite-state machine](/ref/FiniteStateMachine) of the code, in controller canonical form.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.finite_state_machine()
            FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]],
                               outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
        """
        k, nu = self.num_input_bits, self.overall_constraint_length
        next_states = np.empty((2**nu, 2**k), dtype=int)
        outputs = np.empty((2**nu, 2**k), dtype=int)
        for s, x in np.ndindex(2**nu, 2**k):
            initial_state = int_to_bits(s, width=nu)
            u = int_to_bits(x, width=k)
            v, final_state = self.encode_with_state(u, initial_state)
            outputs[s, x] = bits_to_int(v)
            next_states[s, x] = bits_to_int(final_state)
        return FiniteStateMachine(next_states, outputs)

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a given bit sequence, starting from the all-zero state.

        Parameters:
            input: The bit sequence to be encoded. Must be a 1D-array of bits, with length multiple of $k$.

        Returns:
            output: The encoded bit sequence. It is a 1D-array of bits, with length multiple of $n$.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.encode([1, 1, 1, 1])
            array([1, 1, 0, 1, 1, 0, 1, 0])
        """
        nu = self.overall_constraint_length
        output, _ = self.encode_with_state(input, np.zeros(nu, dtype=int))
        return output

    def encode_with_state(
        self,
        input: npt.ArrayLike,
        initial_state: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        """
        Encodes a given bit sequence, starting from a given state.

        Parameters:
            input: The bit sequence to be encoded. Must be a 1D-array of bits, with length multiple of $k$.
            initial_state: The initial state. Must be a 1D-array of length $\\nu$.

        Returns:
            output: The encoded bit sequence. It is a 1D-array of bits, with length multiple of $n$.
            final_state: The final state. It is a 1D-array of length $\\nu$.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.encode_with_state([1, 1, 1, 1], [0, 0])
            (array([1, 1, 0, 1, 1, 0, 1, 0]), array([1, 1]))
            >>> code.encode_with_state([1, 1, 1, 1], [1, 1])
            (array([1, 0, 1, 0, 1, 0, 1, 0]), array([1, 1]))
        """
        n, k = self.num_output_bits, self.num_input_bits
        nu = self.overall_constraint_length
        A_mat, B_mat, C_mat, D_mat = self.state_space_representation()

        input = np.asarray(input).reshape((-1, k))
        state = np.asarray(initial_state)

        if state.ndim != 1:
            raise ValueError("'initial_state' must be a 1D-array")
        if state.size != nu:
            raise ValueError(
                "length of 'initial_state' must be 'overall_constraint_length' "
                f"(expected {nu}, got {state.size})"
            )

        output = np.empty(n * input.size // k, dtype=int)
        for t, u in enumerate(input):
            output[t * n : (t + 1) * n] = (state @ C_mat + u @ D_mat) % 2
            state = (state @ A_mat + u @ B_mat) % 2

        return output, state
