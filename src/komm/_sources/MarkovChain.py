from functools import cache, reduce
from math import gcd

import numpy as np
import numpy.typing as npt

from .._util import global_rng
from .._util.validators import validate_transition_matrix
from ..types import Array1D, Array2D


class MarkovChain:
    r"""
    Finite-state homogeneous discrete-time Markov chain. It is defined by a finite *set of states* $\mathcal{S}$ and a *transition matrix* $P$ over $\mathcal{S}$. Here, for simplicity, the set of states is taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, where $|\mathcal{S}|$ denotes the cardinality of the set of states. For more details, see <cite>YG14, Ch. MCS</cite> and <cite>GS97, Ch. 11</cite>.

    Parameters:
        transition_matrix: The transition matrix $P$ over the set of states $\mathcal{S}$. It must be a $|\mathcal{S}| \times |\mathcal{S}|$-matrix where $P_{i,j}$ is the probability of transitioning from state $i \in \mathcal{S}$ to state $j \in \mathcal{S}$. Its elements must be non-negative and each row must sum to $1$.

    Examples:
        1. Consider the finite-state homogeneous discrete-time Markov chain depicted in the figure below.

            <figure markdown>
            ![Finite-state homogeneous discrete-time Markov chain example.](/fig/markov.svg)
            </figure>

            It has set of states $\mathcal{S} = \\{ 0, 1, 2 \\}$ and transition matrix
            $$
            P = \begin{bmatrix}
                1/2 & 1/4 & 1/4 \\\\
                1/2 &  0  & 1/2 \\\\
                1/4 & 1/4 & 1/2
            \end{bmatrix}.
            $$

                >>> chain = komm.MarkovChain([
                ...     [1/2, 1/4, 1/4],
                ...     [1/2,   0, 1/2],
                ...     [1/4, 1/4, 1/2],
                ... ])
    """

    def __init__(
        self,
        transition_matrix: npt.ArrayLike,
        rng: np.random.Generator | None = None,
    ):
        self._transition_matrix = validate_transition_matrix(
            transition_matrix, square=True
        )
        self._rng = rng or global_rng.get()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.transition_matrix.tolist()})"

    @property
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        return self._transition_matrix

    @property
    def num_states(self) -> int:
        r"""
        The number of states in the Markov chain, $|\mathcal{S}|$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.num_states
            3
        """
        return self._transition_matrix.shape[0]

    @cache
    def accessible_states_from(self, state: int) -> set[int]:
        r"""
        Computes the subset of states that are accessible from a given state. State $j \in \mathcal{S}$ is *accessible from* state $i \in \mathcal{S}$, denoted by $i \to j$, if there exists $n \geq 0$ such that $(P^n)_{i,j} > 0$.

        Parameters:
            state: A state $i \in \mathcal{S}$.

        Returns:
            The subset of all states accessible from state $i$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.accessible_states_from(1)
            {0, 1, 2}

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.accessible_states_from(0)
            {0, 1}
            >>> chain.accessible_states_from(2)
            {2}
        """
        adjacency_matrix = self.transition_matrix > 0
        accessible: set[int] = set()
        stack: list[int] = [state]
        while stack:
            state = stack.pop()
            if state in accessible:
                continue
            accessible.add(state)
            neighbors = np.flatnonzero(adjacency_matrix[state])
            stack.extend(int(j) for j in neighbors if j not in accessible)
        return accessible

    @cache
    def communicating_classes(self) -> list[set[int]]:
        r"""
        Computes the communicating classes of the Markov chain. A *communicating class* is a subset of states such that every state in the class is accessible from every other state in the class. In other words, two states $i, j \in \mathcal{S}$ are in the same communicating class if and only if $i \to j$ and $j \to i$. The set of all communicating classes forms a partition of $\mathcal{S}$.

        Returns:
            A list of all communicating classes in the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.communicating_classes()
            [{0, 1, 2}]

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.communicating_classes()
            [{0, 1}, {2}]
        """
        reach = [self.accessible_states_from(i) for i in range(self.num_states)]
        classes: list[set[int]] = []
        visited: set[int] = set()
        for i in range(self.num_states):
            if i in visited:
                continue
            eq = {j for j in range(self.num_states) if j in reach[i] and i in reach[j]}
            classes.append(eq)
            visited.update(eq)
        return classes

    @cache
    def is_irreducible(self) -> bool:
        r"""
        Returns whether the Markov chain is irreducible. A Markov chain is *irreducible* if $i \to j$ for all $i, j \in \mathcal{S}$. Equivalently, a Markov chain is irreducible if it has only one communicating class.

        Returns:
            `True` if the Markov chain is irreducible, `False` otherwise.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.is_irreducible()
            True

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.is_irreducible()
            False

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.is_irreducible()
            True
        """
        return len(self.communicating_classes()) == 1

    @cache
    def stationary_distribution(self) -> Array1D[np.floating]:
        r"""
        Computes the stationary distribution of an irreducible Markov chain. The *stationary distribution* $\pi$ is a pmf over $\mathcal{S}$ such that $\pi P = \pi$.

        Note:
            This method is only implemented for irreducible chains.

        Returns:
            The stationary distribution $\pi$ of the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.stationary_distribution()
            array([0.4, 0.2, 0.4])

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.stationary_distribution()
            array([0.33333333, 0.33333333, 0.33333333])

            >>> chain = komm.MarkovChain([
            ...     [1, 0, 0],
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ... ])
            >>> chain.stationary_distribution()
            Traceback (most recent call last):
            ...
            NotImplementedError: method is only implemented for irreducible chains
        """
        if not self.is_irreducible():
            raise NotImplementedError(
                "method is only implemented for irreducible chains"
            )
        P, n = self.transition_matrix, self.num_states
        A = np.hstack((P - np.eye(n), np.ones((n, 1))))
        b = np.concatenate((np.zeros(n), [1]))
        pi, *_ = np.linalg.lstsq(A.T, b, rcond=None)
        return pi

    @cache
    def transient_states(self) -> set[int]:
        r"""
        Returns the subset $\mathcal{T} \subseteq \mathcal{S}$ of transient states of the Markov chain. A state $i \in \mathcal{S}$ is *transient* if there exists a state $j \in \mathcal{S}$ such that $i \to j$ but $j \not\to i$.

        Returns:
            The subset $\mathcal{T}$ of transient states in the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.transient_states()
            set()

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2,   0, 1/2],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.transient_states()
            {0, 1}
        """
        reach = [self.accessible_states_from(i) for i in range(self.num_states)]
        transient: set[int] = set()
        for i, r in enumerate(reach):
            for j in r:
                if i not in reach[j]:
                    transient.add(i)
                    break
        return transient

    @cache
    def recurrent_states(self) -> set[int]:
        r"""
        Returns the subset $\mathcal{R} \subseteq \mathcal{S}$ of recurrent states of the Markov chain. A state $i \in \mathcal{S}$ is *recurrent* if it is not transient.

        Returns:
            The subset $\mathcal{R}$ of recurrent states in the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.recurrent_states()
            {0, 1, 2}

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2,   0, 1/2],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.recurrent_states()
            {2}
        """
        return set(range(self.num_states)) - self.transient_states()

    @cache
    def is_regular(self) -> bool:
        r"""
        Returns whether the Markov chain is regular. A Markov chain is *regular* if there exists a positive integer $n$ such that all entries of $P^n$ are positive. A finite-state Markov chain is regular if and only if it is irreducible and aperiodic.

        Returns:
            `True` if the Markov chain is regular, `False` otherwise.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.is_regular()
            True

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.is_regular()
            False

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.is_regular()
            False
        """
        return self.is_irreducible() and self.is_aperiodic()

    def index_of_primitivity(self) -> int:
        r"""
        Computes the index of primitivity of a regular Markov chain. The *index of primitivity* is the smallest positive integer $n$ such that all entries of $P^n$ are positive.

        Notes:
            - This method only applies to regular Markov chains.

        Returns:
            The index of primitivity of the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.index_of_primitivity()
            2

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.index_of_primitivity()
            Traceback (most recent call last):
            ...
            ValueError: chain is not regular
        """
        adjacency_matrix = self.transition_matrix > 0
        Pn = adjacency_matrix.copy()
        max_n = (self.num_states - 1) ** 2 + 1
        for n in range(1, max_n + 1):
            if np.all(Pn):
                return n
            Pn = (Pn @ adjacency_matrix) > 0
        raise ValueError("chain is not regular")

    @cache
    def absorbing_states(self) -> set[int]:
        r"""
        Returns the subset $\mathcal{A} \subseteq \mathcal{S}$ of absorbing states of the Markov chain. A state $i \in \mathcal{S}$ is *absorbing* if $P_{i,i} = 1$.

        Returns:
            The subset $\mathcal{A}$ of absorbing states in the Markov chain.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.absorbing_states()
            set()

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2,   0, 1/2],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.absorbing_states()
            {2}

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.absorbing_states()
            {2}
        """
        return {i for i in range(self.num_states) if self.transition_matrix[i, i] == 1}

    @cache
    def is_absorbing(self) -> bool:
        r"""
        Returns whether the Markov chain is absorbing. A Markov chain is *absorbing* if it has at least one absorbing state and each absorbing state is accessible from at least one transient state.

        Returns:
            `True` if the Markov chain is absorbing, `False` otherwise.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.is_absorbing()
            False

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2,   0, 1/2],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.is_absorbing()
            True

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.is_absorbing()
            False
        """
        absorbing = self.absorbing_states()
        if not absorbing:
            return False
        for i in set(range(self.num_states)) - absorbing:
            if not (absorbing & self.accessible_states_from(i)):
                return False
        return True

    @cache
    def mean_number_of_visits(self) -> Array2D[np.floating]:
        r"""
        Computes the mean number of visits from each transient state to each transient state. This is the expected number of times the chain, starting from transient state $i \in \mathcal{T}$, visits transient state $j \in \mathcal{T}$ before being absorbed.

        Notes:
            - This method only applies to absorbing Markov chains.
            - This corresponds to the *fundamental matrix* $N$ of the Markov chain.


        Returns:
            A $|\mathcal{T}| \times |\mathcal{T}|$-matrix $N$ where entry $N_{i,j}$ is the mean number of visits from $i \in \mathcal{T}$ to $j \in \mathcal{T}$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [  1,   0,   0,   0],
            ...     [1/2,   0, 1/2,   0],
            ...     [  0, 1/2,   0, 1/2],
            ...     [  0,   0,   0,   1],
            ... ])
            >>> chain.mean_number_of_visits()
            array([[1.33333333, 0.66666667],
                   [0.66666667, 1.33333333]])
        """
        if not self.is_absorbing():
            raise ValueError("chain is not absorbing")
        transient = list(self.transient_states())
        P = self.transition_matrix
        Q = P[np.ix_(transient, transient)]
        I = np.eye(len(transient))
        N = np.linalg.inv(I - Q)
        return N

    @cache
    def mean_time_to_absorption(self) -> Array1D[np.floating]:
        r"""
        Computes the mean time to absorption from each transient state. This is the expected number of steps until the chain, starting from transient state $i \in \mathcal{T}$, is absorbed.

        Notes:
            - This method only applies to absorbing Markov chains.
            - This is obtained by adding all the entries in the $i$-th row of the fundamental matrix $N$.

        Returns:
            A $|\mathcal{T}|$-vector $\mathbf{t}$ where $\mathbf{t}_i$ is the mean time to absorption from $i \in \mathcal{T}$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [  1,   0,   0,   0],
            ...     [1/2,   0, 1/2,   0],
            ...     [  0, 1/2,   0, 1/2],
            ...     [  0,   0,   0,   1],
            ... ])
            >>> chain.mean_time_to_absorption()
            array([2., 2.])
        """
        N = self.mean_number_of_visits()
        return N.sum(axis=1)

    @cache
    def absorption_probabilities(self) -> Array2D[np.floating]:
        r"""
        Computes the absorption probabilities from each transient state to each absorbing state. This is the probability of, starting from transient state $i \in \mathcal{T}$, being absorbed in absorbing state $j \in \mathcal{A}$.

        Notes:
            - This method only applies to absorbing Markov chains.
            - This corresponds to the matrix $B = N R$, where $N$ is the fundamental matrix and $R$ is the submatrix of $P$ row-indexed by $\mathcal{T}$ and column-indexed by $\mathcal{A}$.

        Returns:
            A $|\mathcal{T}| \times |\mathcal{A}|$-matrix $B$ where $B_{i,j}$ is the absorption probability from $i \in \mathcal{T}$ to $j \in \mathcal{A}$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [  1,   0,   0,   0],
            ...     [1/2,   0, 1/2,   0],
            ...     [  0, 1/2,   0, 1/2],
            ...     [  0,   0,   0,   1],
            ... ])
            >>> chain.absorption_probabilities()
            array([[0.66666667, 0.33333333],
                   [0.33333333, 0.66666667]])
        """
        N = self.mean_number_of_visits()
        absorbing = list(self.absorbing_states())
        transient = list(self.transient_states())
        P = self.transition_matrix
        R = P[np.ix_(transient, absorbing)]
        B = N @ R
        return B

    @cache
    def period(self, state: int) -> int:
        r"""
        Computes the period of a given state. The *period* of a state $i \in \mathcal{S}$ is the largest integer $d$ such that $(P^n)_{i,i} = 0$ whenever $n$ is not divisible by $d$.

        Parameters:
            state: A state $i \in \mathcal{S}$.

        Returns:
            The period of the state $i$.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.period(1)
            1

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.period(0)
            3
        """
        adjacency_matrix = self.transition_matrix > 0
        Pn = adjacency_matrix.copy()
        periods: list[int] = []
        for n in range(1, self.num_states + 1):
            if Pn[state, state]:
                periods.append(n)
                if reduce(gcd, periods) == 1:
                    return 1
            Pn = (Pn @ adjacency_matrix) > 0
        if not periods:
            return 0
        return reduce(gcd, periods)

    @cache
    def is_aperiodic(self) -> bool:
        r"""
        Returns whether the Markov chain is aperiodic. A Markov chain is *aperiodic* if all states have period $1$.

        Returns:
            `True` if the Markov chain is aperiodic, `False` otherwise.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.is_aperiodic()
            True

            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/2,   0],
            ...     [1/2, 1/2,   0],
            ...     [  0,   0,   1],
            ... ])
            >>> chain.is_aperiodic()
            True

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.is_aperiodic()
            False
        """
        return all(self.period(i) == 1 for i in range(self.num_states))

    def simulate(self, initial_state: int, steps: int) -> Array1D[np.integer]:
        r"""
        Returns random samples from the Markov chain.

        Parameters:
            steps: The number of steps to simulate.

            initial_state: The state $i \in \mathcal{S}$ from which to start the simulation.

        Returns:
            output: A 1D-array of integers in $\mathcal{S}$ representing the states visited during the simulation. The length of the array is equal to `steps`.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [1/2, 1/4, 1/4],
            ...     [1/2,   0, 1/2],
            ...     [1/4, 1/4, 1/2],
            ... ])
            >>> chain.simulate(initial_state=1, steps=10)
            array([1, 2, 1, 2, 2, 0, 2, 2, 2, 0])

            >>> chain = komm.MarkovChain([
            ...     [0, 1, 0],
            ...     [0, 0, 1],
            ...     [1, 0, 0],
            ... ])
            >>> chain.simulate(initial_state=2, steps=10)
            array([2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        """
        P = self.transition_matrix
        state = initial_state
        output = np.empty(steps, dtype=int)
        for t in range(steps):
            output[t] = state
            state = self._rng.choice(self.num_states, p=P[state])
        return output

    def simulate_until_absorption(self, initial_state: int) -> Array1D[np.integer]:
        r"""
        Returns random samples from the Markov chain until an absorbing state is reached.

        Note:
            This method only applies to absorbing Markov chains.

        Parameters:
            initial_state: The state $i \in \mathcal{S}$ from which to start the simulation.

        Returns:
            output: A 1D-array of integers in $\mathcal{S}$ representing the states visited during the simulation.

        Examples:
            >>> chain = komm.MarkovChain([
            ...     [  1,   0,   0,   0],
            ...     [1/2,   0, 1/2,   0],
            ...     [  0, 1/2,   0, 1/2],
            ...     [  0,   0,   0,   1],
            ... ])
            >>> chain.simulate_until_absorption(initial_state=1)
            array([1, 2, 1, 2, 3])
        """
        if not self.is_absorbing():
            raise ValueError("chain is not absorbing")
        P = self.transition_matrix
        state = initial_state
        output = [state]
        while state in self.transient_states():
            state = self._rng.choice(self.num_states, p=P[state])
            output.append(state)
        return np.array(output)
