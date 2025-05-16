from functools import cache, cached_property, reduce
from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction, domain, ring
from .._finite_state_machine import MealyMachine
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.format import format_list_no_quotes as fmt
from .._util.matrices import block_diagonal


class ConvolutionalCode:
    r"""
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* $\mathbf{P}(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $\mathbf{q}(D)$, of length $k$. The element in row $i$ and column $j$ of $\mathbf{P}(D)$ is denoted by $p_{i,j}(D)$, and the element in position $i$ of $\mathbf{q}(D)$ is denoted by $q_i(D)$; they are [binary polynomials](/ref/BinaryPolynomial) in $D$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively. For more details, see <cite>JZ15</cite>, <cite>LC04, Chs. 11, 12</cite>, and <cite>McE98</cite>.

    Parameters:
        feedforward_polynomials: The matrix of feedforward polynomials $\mathbf{P}(D)$, which is a $k \times n$ matrix whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

        feedback_polynomials: The vector of feedback polynomials $\mathbf{q}(D)$, which is a $k$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

    Examples:
        1. Consider the encoder with parameters $(n, k, \sigma) = (2, 1, 6)$ depicted below.

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

        1. Consider the encoder with parameters $(n, k, \sigma) = (3, 2, 7)$ depicted below.

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

        1. Consider the feedback encoder with parameters $(n, k, \sigma) = (2, 1, 4)$ depicted below.

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

    The tables below <cite>LC04, Sec. 12.3</cite> lists optimal convolutional codes with no feedback, for parameters $(n,k) = (2,1)$ and $(n,k) = (3,1)$, and small values of the overall constraint length $\sigma$.

    | Parameters $(n, k, \sigma)$ | Transfer function matrix $\mathbf{G}(D)$ |
    | :-------------------------: | ---------------------------------------- |
    | $(2, 1, 1)$                 | `[[0o1, 0o3]]`                           |
    | $(2, 1, 2)$                 | `[[0o5, 0o7]]`                           |
    | $(2, 1, 3)$                 | `[[0o13, 0o17]]`                         |
    | $(2, 1, 4)$                 | `[[0o27, 0o31]]`                         |
    | $(2, 1, 5)$                 | `[[0o53, 0o75]]`                         |
    | $(2, 1, 6)$                 | `[[0o117, 0o155]]`                       |
    | $(2, 1, 7)$                 | `[[0o247, 0o371]]`                       |
    | $(2, 1, 8)$                 | `[[0o561, 0o753]]`                       |

    | Parameters $(n, k, \sigma)$ | Transfer function matrix $\mathbf{G}(D)$ |
    | :-------------------------: | ---------------------------------------- |
    | $(3, 1, 1)$                 | `[[0o1, 0o3, 0o3]]`                      |
    | $(3, 1, 2)$                 | `[[0o5, 0o7, 0o7]]`                      |
    | $(3, 1, 3)$                 | `[[0o13, 0o15, 0o17]]`                   |
    | $(3, 1, 4)$                 | `[[0o25, 0o33, 0o37]]`                   |
    | $(3, 1, 5)$                 | `[[0o47, 0o53, 0o75]]`                   |
    | $(3, 1, 6)$                 | `[[0o117, 0o127, 0o155]]`                |
    | $(3, 1, 7)$                 | `[[0o255, 0o331, 0o367]]`                |
    | $(3, 1, 8)$                 | `[[0o575, 0o623, 0o727]]`                |
    """

    feedforward_polynomials: list[list[BinaryPolynomial]]
    feedback_polynomials: list[BinaryPolynomial]

    def __init__(
        self,
        feedforward_polynomials: npt.ArrayLike,
        feedback_polynomials: npt.ArrayLike | None = None,
    ) -> None:
        ff = np.asarray(feedforward_polynomials)
        if ff.ndim != 2:
            raise ValueError("feedforward must be a 2-dimensional array")
        if feedback_polynomials is None:
            fb = np.ones(ff.shape[0], dtype=int)
        else:
            fb = np.asarray(feedback_polynomials)
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
        The number of input bits per block, $k$.

        Examples:
            >>> code = komm.ConvolutionalCode(
            ...     feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            ... )
            >>> code.num_input_bits
            2
        """
        return len(self.feedforward_polynomials)

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
        return len(self.feedforward_polynomials[0])

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
        for i, j in product(range(self.num_input_bits), range(self.num_output_bits)):
            p = BinaryPolynomialFraction(self.feedforward_polynomials[i][j])
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
        The *overall constraint length* $\sigma$ of the code, defined by
        $$
            \sigma = \sum_{i \in [0:k)} \nu_i.
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
        The *memory order* $\mu$ of the code, defined by
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

    @cached_property
    def feedforward_taps(self) -> list[npt.NDArray[np.integer]]:
        return [
            np.array([p.coefficients(nu + 1) for p in ps])
            for nu, ps in zip(self.constraint_lengths, self.feedforward_polynomials)
        ]

    @cached_property
    def feedback_taps(self) -> list[npt.NDArray[np.integer]]:
        return [
            q.coefficients(nu + 1)
            for nu, q in zip(self.constraint_lengths, self.feedback_polynomials)
        ]

    @cache
    def free_distance(self) -> int:
        r"""
        Returns the *free distance* $d_\mathrm{free}$ of the code. This is equal to the minimum Hamming weight among all non-zero encoded bit sequences

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.free_distance()
            5
        """
        if self.is_catastrophic():
            raise ValueError(
                "cannot compute free distance when encoder is catastrophic"
            )

        def hamming_weight(y: int, _: int) -> int:
            return y.bit_count()

        fsm = self.finite_state_machine()

        # Start at all-zero state
        metrics = np.full(fsm.num_states, np.inf)
        metrics[0] = 0.0

        _, metrics = fsm.viterbi(
            observed=[-1],  # dummy
            metric_function=hamming_weight,
            initial_metrics=metrics,
        )

        # Block return to all-zero state at the first step
        metrics[0] = np.inf

        while metrics[0] > np.min(metrics[1:]):
            _, metrics = fsm.viterbi(
                observed=[-1],  # dummy
                metric_function=hamming_weight,
                initial_metrics=metrics,
            )

        return int(metrics[0])

    @cache
    def _minors_gcd(self) -> BinaryPolynomialFraction:
        # See [McE98, Sec. 6].
        # Returns the rational polynomial ɑ(D) / β(D) = ɑ'(D) / β'(D).
        # This coincides with the gcd of the k × k minors when G(D) is polynomial
        G = self.transfer_function_matrix
        k, n = G.shape
        denominators: list[BinaryPolynomial] = [x.denominator for row in G for x in row]
        β = reduce(domain.lcm, denominators)
        G_prime: list[list[BinaryPolynomial]] = [
            [(BinaryPolynomialFraction(β) * x).numerator for x in row] for row in G
        ]
        minors: list[BinaryPolynomial] = []
        for js in combinations(range(n), k):
            submatrix = [[row[j] for j in js] for row in G_prime]
            minors.append(ring.determinant(submatrix))
        ɑ = reduce(domain.gcd, minors)
        return BinaryPolynomialFraction(ɑ, β)

    @cache
    def is_catastrophic(self) -> bool:
        """
        Returns whether the encoder is catastrophic. A convolutional encoder is *catastrophic* if there is a finite-weight output sequence generated by an infinite-weight input sequence.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.is_catastrophic()
            False

            >>> code = komm.ConvolutionalCode([[0b11, 0b101]])
            >>> code.is_catastrophic()
            True
        """
        return len(self._minors_gcd().numerator.exponents()) != 1

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
            \mathbf{s}_t & = (s_t^{(0)}, s_t^{(1)}, \ldots, s_t^{(\sigma-1)}),
        \end{aligned}
        $$
        be the input block, output block, and state, respectively, all defined at time instant $t$. Then,
        $$
        \begin{aligned}
            \mathbf{s}\_{t+1} & = \mathbf{s}_t \mathbf{A} + \mathbf{u}_t \mathbf{B}, \\\\
            \mathbf{v}\_{t} & = \mathbf{s}_t \mathbf{C} + \mathbf{u}\_t \mathbf{D},
        \end{aligned}
        $$
        where $\mathbf{A}$ is the $\sigma \times \sigma$ *state matrix*, $\mathbf{B}$ is the $k \times \sigma$ *control matrix*, $\mathbf{C}$ is the $\sigma \times n$ *observation matrix*, and $\mathbf{D}$ is the $k \times n$ *transition matrix*. They are all binary matrices. For more details, see <cite>WBR01</cite>.

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
        alphas = [taps.T for taps in self.feedforward_taps]
        betas = [taps[:, np.newaxis] for taps in self.feedback_taps]

        A_blocks: list[npt.NDArray[np.integer]] = []
        B_blocks: list[npt.NDArray[np.integer]] = []
        C_blocks: list[npt.NDArray[np.integer]] = []
        D_blocks: list[npt.NDArray[np.integer]] = []

        for beta, alpha, nu in zip(alphas, betas, self.constraint_lengths):
            A_blocks.append(np.hstack([alpha[1:], np.eye(nu, nu - 1, dtype=int)]))
            B_blocks.append(np.eye(1, nu, dtype=int))
            C_blocks.append(beta[1:] ^ alpha[1:] * beta[0])
            D_blocks.append(beta[0])

        A_mat = block_diagonal(A_blocks)
        B_mat = block_diagonal(B_blocks)
        C_mat = np.vstack(C_blocks)
        D_mat = np.vstack(D_blocks)

        return A_mat, B_mat, C_mat, D_mat

    @cache
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Returns the [finite-state (Mealy) machine](/ref/MealyMachine) of the code, in controller canonical form.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.finite_state_machine()
            MealyMachine(transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
                         outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
        """
        k, σ = self.num_input_bits, self.overall_constraint_length
        transitions = np.empty((2**σ, 2**k), dtype=int)
        outputs = np.empty((2**σ, 2**k), dtype=int)
        for s, x in np.ndindex(2**σ, 2**k):
            initial_state = int_to_bits(s, width=σ)
            u = int_to_bits(x, width=k)
            v, final_state = self.encode_with_state(u, initial_state)
            transitions[s, x] = bits_to_int(final_state)
            outputs[s, x] = bits_to_int(v)
        return MealyMachine(transitions, outputs)

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
        σ = self.overall_constraint_length
        output, _ = self.encode_with_state(input, np.zeros(σ, dtype=int))
        return output

    def encode_with_state(
        self,
        input: npt.ArrayLike,
        initial_state: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        r"""
        Encodes a given bit sequence, starting from a given state.

        Parameters:
            input: The bit sequence to be encoded. Must be a 1D-array of bits, with length multiple of $k$.
            initial_state: The initial state. Must be a 1D-array of length $\sigma$.

        Returns:
            output: The encoded bit sequence. It is a 1D-array of bits, with length multiple of $n$.
            final_state: The final state. It is a 1D-array of length $\sigma$.

        Examples:
            >>> code = komm.ConvolutionalCode([[0b111, 0b101]])
            >>> code.encode_with_state([1, 1, 1, 1], [0, 0])
            (array([1, 1, 0, 1, 1, 0, 1, 0]), array([1, 1]))
            >>> code.encode_with_state([1, 1, 1, 1], [1, 1])
            (array([1, 0, 1, 0, 1, 0, 1, 0]), array([1, 1]))
        """
        n, k = self.num_output_bits, self.num_input_bits
        σ = self.overall_constraint_length
        A_mat, B_mat, C_mat, D_mat = self.state_space_representation()

        input = np.asarray(input).reshape((-1, k))
        state = np.asarray(initial_state)

        if state.ndim != 1:
            raise ValueError("'initial_state' must be a 1D-array")
        if state.size != σ:
            raise ValueError(
                "length of 'initial_state' must be 'overall_constraint_length' "
                f"(expected {σ}, got {state.size})"
            )

        output = np.empty(n * input.size // k, dtype=int)
        for t, u in enumerate(input):
            output[t * n : (t + 1) * n] = (state @ C_mat + u @ D_mat) % 2
            state = (state @ A_mat + u @ B_mat) % 2

        return output, state
