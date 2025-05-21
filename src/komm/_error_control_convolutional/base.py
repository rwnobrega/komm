from abc import ABC, abstractmethod
from functools import cache, cached_property, reduce
from heapq import heappop, heappush
from itertools import combinations

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction, domain, ring
from .._finite_state_machine.MealyMachine import MealyMachine
from .._util.bit_operations import bits_to_int, int_to_bits


class ConvolutionalCode(ABC):
    @cached_property
    @abstractmethod
    def num_input_bits(self) -> int:
        r"""
        The number of input bits per block, $k$.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def num_output_bits(self) -> int:
        r"""
        The number of output bits per block, $n$.
        """
        raise NotImplementedError

    @cached_property
    def degree(self) -> int:
        r"""
        The degree of the encoder, $\sigma$.
        """
        raise NotImplementedError

    @cache
    @abstractmethod
    def state_space_representation(
        self,
    ) -> tuple[
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
    ]:
        r"""
        Returns the *state-space representation* of the encoder. Let
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
            state_matrix: The state matrix $\mathbf{A}$ of the encoder.
            control_matrix: The control matrix $\mathbf{B}$ of the encoder.
            observation_matrix: The observation matrix $\mathbf{C}$ of the encoder.
            transition_matrix: The transition matrix $\mathbf{D}$ of the encoder.
        """
        raise NotImplementedError

    @cache
    @abstractmethod
    def generator_matrix(self) -> npt.NDArray[np.object_]:
        r"""
        Returns the *transform-domain generator matrix* (also known as *transfer function matrix*) $\mathbf{G}(D)$ of the encoder. This is a $k \times n$ array of [binary polynomial fractions](/ref/BinaryPolynomialFraction).
        """
        raise NotImplementedError

    @cache
    @abstractmethod
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Returns the [finite-state (Mealy) machine](/ref/MealyMachine) of the encoder.
        """
        k, σ = self.num_input_bits, self.degree
        transitions = np.empty((2**σ, 2**k), dtype=int)
        outputs = np.empty((2**σ, 2**k), dtype=int)
        for s, x in np.ndindex(2**σ, 2**k):
            initial_state = int_to_bits(s, width=σ)
            u = int_to_bits(x, width=k)
            v, final_state = self.encode_with_state(u, initial_state)
            transitions[s, x] = bits_to_int(final_state)
            outputs[s, x] = bits_to_int(v)
        return MealyMachine(transitions, outputs)

    @cache
    def _minors_gcd(self) -> BinaryPolynomialFraction:
        # See [McE98, Sec. 6].
        # Returns the rational polynomial ɑ(D) / β(D) = ɑ'(D) / β'(D).
        # This coincides with the gcd of the k × k minors when G(D) is polynomial
        G_mat = self.generator_matrix()
        k, n = G_mat.shape
        denominators: list[BinaryPolynomial] = [
            x.denominator for row in G_mat for x in row
        ]
        β = reduce(domain.lcm, denominators)
        G_prime: list[list[BinaryPolynomial]] = [
            [(BinaryPolynomialFraction(β) * x).numerator for x in row] for row in G_mat
        ]
        minors: list[BinaryPolynomial] = []
        for js in combinations(range(n), k):
            submatrix = [[row[j] for j in js] for row in G_prime]
            minors.append(ring.determinant(submatrix))
        ɑ = reduce(domain.gcd, minors)
        return BinaryPolynomialFraction(ɑ, β)

    @cache
    @abstractmethod
    def is_catastrophic(self) -> bool:
        """
        Returns whether the encoder is catastrophic. A convolutional encoder is *catastrophic* if there exists an infinite-weight input sequence that generates a finite-weight output sequence.
        """
        minors_gcd = self._minors_gcd()
        return len(minors_gcd.numerator.exponents()) != 1

    @cache
    @abstractmethod
    def free_distance(self) -> int:
        r"""
        Returns the *free distance* $d_\mathrm{free}$ of the code. This is equal to the minimum Hamming weight among all possible non-zero output sequences.
        """
        fsm = self.finite_state_machine()
        heap = [(0, 0)]  # (weight, state)
        best = [np.inf] * fsm.num_states
        while heap:
            w, s = heappop(heap)
            if s == 0 and w > 0:
                return int(w)
            for x in range(fsm.num_input_symbols):
                s1 = fsm.transitions[s][x]
                w1 = w + fsm.outputs[s][x].bit_count()
                if w1 < best[s1]:
                    best[s1] = w1
                    heappush(heap, (w1, s1))
                elif s1 == 0 and w1 > 0:
                    heappush(heap, (w1, 0))
        raise RuntimeError

    @abstractmethod
    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a given bit sequence, starting from the all-zero state.

        Parameters:
            input: The bit sequence to be encoded. Must be a 1D-array of bits, with length multiple of $k$.

        Returns:
            output: The encoded bit sequence. It is a 1D-array of bits, with length multiple of $n$.
        """
        σ = self.degree
        output, _ = self.encode_with_state(input, np.zeros(σ, dtype=int))
        return output

    @abstractmethod
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
        """
        n, k, σ = self.num_output_bits, self.num_input_bits, self.degree
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
