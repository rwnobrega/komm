from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import FiniteStateMachine
from .._util.bit_operations import binlist2int, int2binlist


class ConvolutionalCode:
    r"""
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* $P(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $q(D)$, of length $k$. The element in row $i$ and column $j$ of $P(D)$ is denoted by $p_{i,j}(D)$, and the element in position $i$ of $q(D)$ is denoted by $q_i(D)$; they are [binary polynomials](/ref/BinaryPolynomial) in $D$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively.

    <h2>Transfer function matrix</h2>

    The *transfer function matrix* (also known as *transform-domain generator matrix*) $G(D)$ of the convolutional code, of shape $k \times n$, is such that the element in row $i$ and column $j$ is given by
    $$
        g_{i,j}(D) = \frac{p_{i,j}(D)}{q_{i}(D)},
    $$
    for $i \in [0 : k)$ and $j \in [0 : n)$.

    <h2>Constraint lengths and related parameters</h2>

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

    <h2>Space-state representation</h2>

    A convolutional code may also be described via the *space-state representation*. Let $\mathbf{u}_t = (u_t^{(0)}, u_t^{(1)}, \ldots, u_t^{(k-1)})$ be the input block, $\mathbf{v}_t = (v_t^{(0)}, v_t^{(1)}, \ldots, v_t^{(n-1)})$ be the output block, and $\mathbf{s}_t = (s_t^{(0)}, s_t^{(1)}, \ldots, s_t^{(\nu-1)})$ be the state, all defined at time instant $t$. Then,
    $$
    \begin{aligned}
        \mathbf{s}\_{t+1} & = \mathbf{s}_t A + \mathbf{u}_t B, \\\\
        \mathbf{v}\_{t} & = \mathbf{s}_t C + \mathbf{u}\_t D,
    \end{aligned}
    $$
    where $A$ is the $\nu \times \nu$ *state matrix*, $B$ is the $k \times \nu$ *control matrix*, $C$ is the $\nu \times n$ *observation matrix*, and $D$ is the $k \times n$ *transition matrix*.

    <h2>Tables of convolutional codes</h2>

    The tables below <cite>LC04, Sec. 12.3</cite> lists optimal convolutional codes with no feedback, for parameters $(n,k) = (2,1)$ and $(n,k) = (3,1)$, and small values of the overall constraint length $\nu$.

    | Parameters $(n, k, \nu)$ | Transfer function matrix $G(D)$ |
    | :----------------------: | ------------------------------- |
    | $(2, 1, 1)$              | `[[0o1, 0o3]]`                  |
    | $(2, 1, 2)$              | `[[0o5, 0o7]]`                  |
    | $(2, 1, 3)$              | `[[0o13, 0o17]]`                |
    | $(2, 1, 4)$              | `[[0o27, 0o31]]`                |
    | $(2, 1, 5)$              | `[[0o53, 0o75]]`                |
    | $(2, 1, 6)$              | `[[0o117, 0o155]]`              |
    | $(2, 1, 7)$              | `[[0o247, 0o371]]`              |
    | $(2, 1, 8)$              | `[[0o561, 0o753]]`              |

    | Parameters $(n, k, \nu)$ | Transfer function matrix $G(D)$ |
    | :----------------------: | ------------------------------- |
    | $(3, 1, 1)$              | `[[0o1, 0o3, 0o3]]`             |
    | $(3, 1, 2)$              | `[[0o5, 0o7, 0o7]]`             |
    | $(3, 1, 3)$              | `[[0o13, 0o15, 0o17]]`          |
    | $(3, 1, 4)$              | `[[0o25, 0o33, 0o37]]`          |
    | $(3, 1, 5)$              | `[[0o47, 0o53, 0o75]]`          |
    | $(3, 1, 6)$              | `[[0o117, 0o127, 0o155]]`       |
    | $(3, 1, 7)$              | `[[0o255, 0o331, 0o367]]`       |
    | $(3, 1, 8)$              | `[[0o575, 0o623, 0o727]]`       |

    For more details, see <cite>JZ15</cite> and <cite>LC04, Chs. 11, 12</cite>.
    """

    def __init__(
        self,
        feedforward_polynomials: npt.ArrayLike,
        feedback_polynomials: Optional[npt.ArrayLike] = None,
    ) -> None:
        r"""
        Constructor for the class.

        Parameters:
            feedforward_polynomials (Array2D[BinaryPolynomial, int]): The matrix of feedforward polynomials $P(D)$, which is a $k \times n$ matrix whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

            feedback_polynomials (Optional[Array1D[BinaryPolynomial, int]]): The vector of feedback polynomials $q(D)$, which is a $k$-vector whose entries are either [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

        Examples:
            The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 6)$; its transfer function matrix is given by
            $$
                G(D) =
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

            The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (3, 2, 7)$; its transfer function matrix is given by
            $$
                G(D) =
                \begin{bmatrix}
                    D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\\\
                    0  &  D^3 + D  &  D^3 + D^2 + 1
                \end{bmatrix},
            $$
            yielding `feedforward_polynomials = [[0b11001, 0b10111, 0b00000], [0b0000, 0b1010, 0b1101]] = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]] = [[25, 23, 0], [0, 10, 13]]`.

            <figure markdown>
              ![Convolutional encoder for (3, 2, 7) code.](/figures/cc_3_2_7.svg)
            </figure>

            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]])
            >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
            (3, 2, 7)

            The convolutional code with feedback encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 4)$; its transfer function matrix is given by
            $$
                G(D) =
                \begin{bmatrix}
                    1  &  \dfrac{D^4 + D^3 + 1}{D^4 + D^2 + D + 1}
                \end{bmatrix},
            $$
            yielding `feedforward_polynomials = [[0b10111, 0b11001]] = [[0o27, 0o31]] = [[23, 25]]` and `feedback_polynomials = [0o27]`.

            <figure markdown>
              ![Convolutional encoder for (2, 1, 4) feedback code.](/figures/cc_2_1_4_fb.svg)
            </figure>

            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o27, 0o31]], feedback_polynomials=[0o27])
            >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
            (2, 1, 4)
        """
        self.feedforward_polynomials: npt.NDArray[np.object_]
        self.feedback_polynomials: npt.NDArray[np.object_]
        vecBinaryPolynomial = np.vectorize(BinaryPolynomial)

        self.feedforward_polynomials = vecBinaryPolynomial(feedforward_polynomials)

        if feedback_polynomials is None:
            k = self.feedforward_polynomials.shape[0]
            self.feedback_polynomials = vecBinaryPolynomial([0b1] * k)
            self._constructed_from = "no_feedback_polynomials"
        else:
            self.feedback_polynomials = vecBinaryPolynomial(feedback_polynomials)
            self._constructed_from = "feedback_polynomials"

        self._setup_finite_state_machine_direct_form()
        self._setup_space_state_representation()

    def __repr__(self) -> str:
        def vec_str(arr: npt.NDArray[np.object_]) -> str:
            return str(np.vectorize(str)(arr).tolist()).replace("'", "")

        args = f"feedforward_polynomials={vec_str(self.feedforward_polynomials)}"
        if self._constructed_from == "feedback_polynomials":
            args += f", feedback_polynomials={vec_str(self.feedback_polynomials)}"
        return "{}({})".format(self.__class__.__name__, args)

    def _setup_finite_state_machine_direct_form(self) -> None:
        n, k, nu = (
            self.num_output_bits,
            self.num_input_bits,
            self.overall_constraint_length,
        )

        x_indices = np.concatenate(([0], np.cumsum(self.constraint_lengths + 1)[:-1]))
        s_indices = np.setdiff1d(np.arange(k + nu, dtype=int), x_indices)

        feedforward_taps: list[npt.NDArray[np.object_]] = []
        for j in range(n):
            taps = np.concatenate([
                self.feedforward_polynomials[i, j].exponents() + x_indices[i]
                for i in range(k)
            ])
            feedforward_taps.append(taps)

        feedback_taps: list[npt.NDArray[np.object_]] = []
        for i in range(k):
            taps = (
                BinaryPolynomial(0b1) + self.feedback_polynomials[i]
            ).exponents() + x_indices[i]
            feedback_taps.append(taps)

        bits = np.empty(k + nu, dtype=int)
        next_states = np.empty((2**nu, 2**k), dtype=int)
        outputs = np.empty((2**nu, 2**k), dtype=int)

        for s, x in np.ndindex(2**nu, 2**k):
            bits[s_indices] = int2binlist(s, width=nu)
            bits[x_indices] = int2binlist(x, width=k)
            bits[x_indices] ^= [
                np.count_nonzero(bits[feedback_taps[i]]) % 2 for i in range(k)
            ]

            next_state_bits = bits[s_indices - 1]
            output_bits = [
                np.count_nonzero(bits[feedforward_taps[j]]) % 2 for j in range(n)
            ]

            next_states[s, x] = binlist2int(next_state_bits)
            outputs[s, x] = binlist2int(output_bits)

        self._finite_state_machine = FiniteStateMachine(
            next_states=next_states, outputs=outputs
        )

    def _setup_space_state_representation(self) -> None:
        k, n, nu = (
            self.num_input_bits,
            self.num_output_bits,
            self.overall_constraint_length,
        )

        self._state_matrix = np.empty((nu, nu), dtype=int)
        self._observation_matrix = np.empty((nu, n), dtype=int)
        for i in range(nu):
            s0 = 2**i
            s1 = self._finite_state_machine.next_states[s0, 0]
            y = self._finite_state_machine.outputs[s0, 0]
            self._state_matrix[i, :] = int2binlist(s1, width=nu)
            self._observation_matrix[i, :] = int2binlist(y, width=n)

        self._control_matrix = np.empty((k, nu), dtype=int)
        self._transition_matrix = np.empty((k, n), dtype=int)
        for i in range(k):
            x = 2**i
            s1 = self._finite_state_machine.next_states[0, x]
            y = self._finite_state_machine.outputs[0, x]
            self._control_matrix[i, :] = int2binlist(s1, width=nu)
            self._transition_matrix[i, :] = int2binlist(y, width=n)

    @property
    def num_input_bits(self) -> int:
        r"""
        The number of input bits per block, $k$.
        """
        return self.feedforward_polynomials.shape[0]

    @property
    def num_output_bits(self) -> int:
        r"""
        The number of output bits per block, $n$.
        """
        return self.feedforward_polynomials.shape[1]

    @cached_property
    def constraint_lengths(self) -> npt.NDArray[np.int_]:
        r"""
        The constraint lengths $\nu_i$ of the code, for $i \in [0 : k)$. This is a $k$-array of integers.
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
        """
        return int(np.sum(self.constraint_lengths))

    @cached_property
    def memory_order(self) -> int:
        r"""
        The memory order $\mu$ of the code.
        """
        return int(np.max(self.constraint_lengths))

    @cached_property
    def transfer_function_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        The transfer function matrix $G(D)$ of the code. This is a $k \times n$ array of [binary polynomial fractions](/ref/BinaryPolynomialFraction).
        """
        transfer_function_matrix = np.empty_like(self.feedforward_polynomials)
        for i, j in np.ndindex(self.feedforward_polynomials.shape):
            p = BinaryPolynomialFraction(self.feedforward_polynomials[i, j])
            q = BinaryPolynomialFraction(self.feedback_polynomials[i])
            transfer_function_matrix[i, j] = p / q
        return transfer_function_matrix

    @property
    def finite_state_machine(self) -> FiniteStateMachine:
        r"""
        The finite-state machine of the code.
        """
        return self._finite_state_machine

    @property
    def state_matrix(self) -> npt.NDArray[np.int_]:
        r"""
        The state matrix $A$ of the state-space representation. This is a $\nu \times \nu$ array of integers in $\\{ 0, 1 \\}$.
        """
        return self._state_matrix

    @property
    def control_matrix(self) -> npt.NDArray[np.int_]:
        r"""
        The control matrix $B$ of the state-space representation. This is a $k \times \nu$ array of integers in $\\{ 0, 1 \\}$.
        """
        return self._control_matrix

    @property
    def observation_matrix(self) -> npt.NDArray[np.int_]:
        r"""
        The observation matrix $C$ of the state-space representation. This is a $\nu \times n$ array of integers in $\\{ 0, 1 \\}$.
        """
        return self._observation_matrix

    @property
    def transition_matrix(self) -> npt.NDArray[np.int_]:
        r"""
        The transition matrix $D$ of the state-space representation. This is a $k \times n$ array of integers in $\\{ 0, 1 \\}$.
        """
        return self._transition_matrix
