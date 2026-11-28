from abc import ABC, abstractmethod
from collections import Counter
from functools import cache, cached_property, reduce
from heapq import heappop, heappush
from itertools import combinations

import numpy as np
import numpy.typing as npt

from .._algebra import domain, ring
from .._algebra.BinaryPolynomial import BinaryPolynomial
from .._algebra.BinaryPolynomialFraction import BinaryPolynomialFraction
from .._finite_state_machine.MealyMachine import MealyMachine
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.matrices import invariant_factors, matmul
from .._util.validators import validate_binary_array, validate_nonnegative_integer
from ..types import Array1D, Array2D


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
    @abstractmethod
    def degree(self) -> int:
        r"""
        The *degree* $\sigma$ of the encoder. This corresponds to the number of delay elements in the encoder realization.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def generator_matrix(self) -> Array2D[np.object_]:
        r"""
        Returns the (transform-domain) *generator matrix* (also known as *transfer function matrix*) $G(D)$ of the encoder. This is a $k \times n$ array of [binary polynomial fractions](/ref/BinaryPolynomialFraction).
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def constraint_lengths(self) -> Array1D[np.integer]:
        r"""
        The *constraint lengths* $\nu_i$ of the encoder, for $i \in [0 : k)$. These are defined by
        $$
            \nu_i = \max \\{ \deg p_{i,0}(D), \deg p_{i,1}(D), \ldots, \deg p_{i,n-1}(D), \deg q_i(D) \\},
        $$
        where $p_{i,j}(D) / q_i(D)$ are the entries of the $i$-th row of the generator matrix $G(D)$, satisfying
        $$
            \gcd(p_{i,0}(D), p_{i,1}(D), \ldots, p_{i,n-1}(D), q_i(D)) = 1.
        $$
        This is a $k$-array of integers.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def overall_constraint_length(self) -> int:
        r"""
        The *overall constraint length* $\nu$ of the encoder, defined by
        $$
            \nu = \sum_{i \in [0 : k)} \nu_i,
        $$
        where $\nu_i$ are the constraint lengths of the encoder.
        """
        return int(np.sum(self.constraint_lengths))

    @cached_property
    @abstractmethod
    def memory_order(self) -> int:
        r"""
        The *memory order* $\mu$ of the encoder. It is given by
        $$
            \mu = \max_{i \in [0 : k)} \nu_i,
        $$
        where $\nu_i$ are the constraint lengths of the encoder.
        """
        return int(np.max(self.constraint_lengths))

    @cache
    @abstractmethod
    def state_space_representation(
        self,
    ) -> tuple[
        Array2D[np.integer],
        Array2D[np.integer],
        Array2D[np.integer],
        Array2D[np.integer],
    ]:
        r"""
        Returns the matrices $(\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D})$ corresponding to the state-space representation of the encoder realization. The *state-space representation* of the encoder is given by
        $$
        \begin{aligned}
            s_{t+1} & = s_t \mathbf{A} + u_t \mathbf{B}, \\\\
            v_t & = s_t \mathbf{C} + u_t \mathbf{D},
        \end{aligned}
        $$
        where

        - $u_t \in \mathbb{B}^k$ is the *input block*,
        - $v_t \in \mathbb{B}^n$ is the *output block*,
        - $s_t \in \mathbb{B}^\sigma$ is the *state*,
        - $\mathbf{A} \in \mathbb{B}^{\sigma \times \sigma}$ is the *state matrix*,
        - $\mathbf{B} \in \mathbb{B}^{k \times \sigma}$ is the *control matrix*,
        - $\mathbf{C} \in \mathbb{B}^{\sigma \times n}$ is the *observation matrix*,
        - $\mathbf{D} \in \mathbb{B}^{k \times n}$ is the *transition matrix*.

        Returns:
            state_matrix: The state matrix $\mathbf{A}$ of the encoder.
            control_matrix: The control matrix $\mathbf{B}$ of the encoder.
            observation_matrix: The observation matrix $\mathbf{C}$ of the encoder.
            transition_matrix: The transition matrix $\mathbf{D}$ of the encoder.
        """
        raise NotImplementedError

    @cache
    @abstractmethod
    def finite_state_machine(self) -> MealyMachine:
        r"""
        Returns the [finite-state (Mealy) machine](/ref/MealyMachine) of the encoder.

        In the machine returned by this method, the states, inputs, and outputs are the integers whose binary representations (LSB-first) are the state, the input block, and the output block of [`encode_with_state`](./#encode_with_state).
        """
        n, k, σ = self.num_output_bits, self.num_input_bits, self.degree
        # All (state, input) pairs in one step
        states = int_to_bits(range(2**σ), width=σ).reshape(2**σ, 1, σ)
        inputs = int_to_bits(range(2**k), width=k).reshape(1, 2**k, k)
        inputs = np.broadcast_to(inputs, (2**σ, 2**k, k))
        v, final_states = self.encode_with_state(inputs, states)
        transitions = np.zeros((2**σ, 2**k), dtype=int)
        if σ > 0:  # Memoryless encoder has a single state
            transitions = bits_to_int(final_states, width=σ).reshape(2**σ, 2**k)
        outputs = bits_to_int(v, width=n).reshape(2**σ, 2**k)
        return MealyMachine(transitions, outputs)

    @cache
    def _minors_gcd(self) -> BinaryPolynomial:
        # We keep this method for testing purposes.
        G_mat = self.generator_matrix
        k, n = G_mat.shape
        if any(x.denominator != 0b1 for row in G_mat for x in row):
            raise ValueError("this method only works with polynomial matrices")
        G_poly: list[list[BinaryPolynomial]] = [
            [x.numerator for x in row] for row in G_mat
        ]
        minors: list[BinaryPolynomial] = []
        for js in combinations(range(n), k):
            submatrix = [[row[j] for j in js] for row in G_poly]
            minors.append(ring.determinant(submatrix))
        return reduce(domain.gcd, minors)

    @cache
    @abstractmethod
    def is_catastrophic(self) -> bool:
        """
        Returns whether the encoder is catastrophic. A convolutional encoder is *catastrophic* if there exists an infinite-weight input sequence that generates a finite-weight output sequence.
        """
        # See [JZ15, Secs. 2.2 and 2.3], in particular Theorem 2.10.
        G_mat = self.generator_matrix
        denominators = [x.denominator for row in G_mat for x in row]
        q = reduce(domain.lcm, denominators)
        G_prime = np.array([
            [(BinaryPolynomialFraction(q) * x).numerator for x in row] for row in G_mat
        ])
        factors = invariant_factors(G_prime)
        ɑ_k = BinaryPolynomialFraction(factors[-1], q).numerator
        return len(ɑ_k.exponents()) != 1

    @cache
    @abstractmethod
    def path_weight_distribution(self, max_weight: int) -> Array2D[np.integer]:
        r"""
        Returns the path weight distribution of the code, up to a maximum output weight. This is the table $A_{i,j}$, in which $A_{i,j}$ is the number of paths in the state diagram that leave the all-zero state and first return to it with input weight $i$ and output weight $j$. These are the coefficients of the *weight enumerator*
        $$
            A(U, V) = \sum_{i,j} A_{i,j} U^i V^j.
        $$
        The number of paths with output weight $j$ is $A_j = \sum_i A_{i,j}$, and their total input weight is $B_j = \sum_i i A_{i,j}$. The least $j$ with $A_j > 0$ is the [free distance](./#free_distance).

        This method keeps only the paths with output weight $j \leq J$. The table has one column for each $j \in [0 : J]$, and one row for each input weight from $0$ up to the largest among the paths kept.

        Parameters:
            max_weight: The maximum output weight $J$. Must be a non-negative integer.

        Returns:
            distribution: The table $A_{i,j}$, with the input weight $i$ along the rows and the output weight $j$ along the columns.

        Raises:
            ValueError: If the state diagram has a cycle of zero output weight (as in catastrophic encoders), or if some $B_j$ exceeds $2^{63} - 1$.
        """
        max_weight = validate_nonnegative_integer(max_weight)
        fsm = self.finite_state_machine()
        nxt: list[list[int]] = fsm.transitions.tolist()
        out: list[list[int]] = fsm.outputs.tolist()
        inputs = range(fsm.num_input_symbols)
        w_in = [x.bit_count() for x in inputs]
        w_out = [[y.bit_count() for y in row] for row in out]

        # Nonzero states reachable from zero
        reached, queue = {0}, [0]
        for s in queue:
            for s1 in nxt[s]:
                if s1 not in reached:
                    reached.add(s1)
                    queue.append(s1)
        states = queue[1:]

        # Topological order of zero-weight branches
        zero = {
            s: [s1 for s1, w in zip(nxt[s], w_out[s]) if w == 0 and s1 != 0]
            for s in states
        }
        indegree = Counter(s1 for s in states for s1 in zero[s])
        order = [s for s in states if indegree[s] == 0]
        for s in order:
            for s1 in zero[s]:
                indegree[s1] -= 1
                if indegree[s1] == 0:
                    order.append(s1)
        if not len(order) == len(states):
            raise ValueError("state diagram has a cycle of zero output weight")

        # paths[j][s]: input weights of paths at s
        paths: list[dict[int, Counter[int]]] = [{} for _ in range(max_weight + 1)]

        def extend(j: int, s: int, x: int, counts: Counter[int]) -> None:
            j1 = j + w_out[s][x]
            if j1 <= max_weight:
                target = paths[j1].setdefault(nxt[s][x], Counter())
                for i, count in counts.items():
                    target[i + w_in[x]] += count

        # Within each layer, follow topological order
        for x in inputs[1:]:
            extend(0, 0, x, Counter({0: 1}))
        for j in range(max_weight + 1):
            for s in order:
                if s in paths[j]:
                    for x in inputs:
                        extend(j, s, x, paths[j][s])

        arrivals = [paths[j].get(0, Counter()) for j in range(max_weight + 1)]
        if not max(sum(i * c for i, c in a.items()) for a in arrivals) < 2**63:
            raise ValueError("'max_weight' is too large for 64-bit counts")
        num_rows = max((i for a in arrivals for i in a), default=0) + 1
        return np.array([[a[i] for a in arrivals] for i in range(num_rows)])

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
        Encodes bit sequences, starting from the all-zero state.

        Parameters:
            input: The bit sequence(s) to be encoded. Can be either a single sequence whose length is a multiple of $k$, or a multidimensional array where the last dimension is a multiple of $k$.

        Returns:
            output: The encoded bit sequence(s). Has the same shape as the input, but with the last dimension expanded by a factor of $n / k$.
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
        Encodes bit sequences, starting from given states.

        Parameters:
            input: The bit sequence(s) to be encoded. Can be either a single sequence whose length is a multiple of $k$, or a multidimensional array where the last dimension is a multiple of $k$.

            initial_state: The initial state(s). Its elements must be `0` or `1`. Must be either a 1D-array of length $\sigma$, or an array with the leading dimensions of the input and length $\sigma$ in the last dimension.

        Returns:
            output: The encoded bit sequence(s). Has the same shape as the input, but with the last dimension expanded by a factor of $n / k$.

            final_state: The final state(s). Has the same shape as the input, but with the last dimension replaced by $\sigma$.
        """
        n, k, σ = self.num_output_bits, self.num_input_bits, self.degree
        A_mat, B_mat, C_mat, D_mat = self.state_space_representation()

        input = np.atleast_1d(validate_binary_array(input))
        if not input.shape[-1] % k == 0:
            raise ValueError(
                f"last dimension of 'input' must be a multiple of {k}"
                f" (got {input.shape[-1]})"
            )
        u = input.reshape(*input.shape[:-1], -1, k)
        state = np.atleast_1d(validate_binary_array(initial_state))

        if state.shape[-1] != σ:
            raise ValueError(
                "last dimension of 'initial_state' must be 'degree' "
                f"(expected {σ}, got {state.shape[-1]})"
            )

        # Per step, [v_t, s_{t+1}] = [s_t, u_t] M
        M_mat = np.block([[C_mat, A_mat], [D_mat, B_mat]])
        su = np.empty((*u.shape[:-2], σ + k), dtype=int)
        su[..., :σ] = state
        output = np.empty((*u.shape[:-1], n), dtype=int)
        for t in range(u.shape[-2]):
            su[..., σ:] = u[..., t, :]
            vs = matmul(su, M_mat)
            output[..., t, :] = vs[..., :n]
            su[..., :σ] = vs[..., n:]

        return output.reshape(*input.shape[:-1], -1), su[..., :σ]
