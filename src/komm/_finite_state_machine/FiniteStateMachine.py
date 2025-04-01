from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt


class MetricMemory(TypedDict):
    paths: npt.NDArray[np.integer]
    metrics: npt.NDArray[np.floating]


Z = TypeVar("Z")
MetricFunction = Callable[[int, Z], float]


class FiniteStateMachine:
    r"""
    Finite-state machine (Mealy machine). It is defined by a *set of states* $\mathcal{S}$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, and a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S} \times \mathcal{Y}$. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as $\mathcal{S} = \\{ 0, 1, \ldots, |\mathcal{S}| - 1 \\}$, $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$, and $\mathcal{Y} = \\{ 0, 1, \ldots, |\mathcal{Y}| - 1 \\}$, respectively.

    For example, consider the finite-state machine whose state diagram depicted in the figure below.

    <figure markdown>
      ![Finite-state machine (Mealy machine) example.](/figures/mealy.svg)
    </figure>

    It has set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$, and transition function $T$ given by the table below.

    | State | Input | Next state | Output |
    | :---: | :---: | :--------: | :----: |
    | $0$   | $0$   | $0$        | $0$    |
    | $0$   | $1$   | $1$        | $3$    |
    | $1$   | $0$   | $2$        | $1$    |
    | $1$   | $1$   | $3$        | $2$    |
    | $2$   | $0$   | $0$        | $3$    |
    | $2$   | $1$   | $1$        | $0$    |
    | $3$   | $0$   | $2$        | $2$    |
    | $3$   | $1$   | $3$        | $1$    |

    Parameters:
        next_states: The matrix of next states of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ should be the next state of the machine (an element in $\mathcal{S}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$.

        outputs: The matrix of outputs of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ should be the output of the machine (an element in $\mathcal{Y}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$.

    Examples:
        >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
    """

    def __init__(self, next_states: npt.ArrayLike, outputs: npt.ArrayLike):
        self.next_states = np.asarray(next_states)
        self.outputs = np.asarray(outputs)

    def __repr__(self) -> str:
        args = ", ".join([
            f"next_states={self.next_states.tolist()}",
            f"outputs={self.outputs.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def num_states(self) -> int:
        r"""
        The number of states of the machine.
        """
        return self.next_states.shape[0]

    @cached_property
    def num_input_symbols(self) -> int:
        r"""
        The size (cardinality) of the input alphabet $\mathcal{X}$.
        """
        return self.next_states.shape[1]

    @cached_property
    def num_output_symbols(self) -> int:
        r"""
        The size (cardinality) of the output alphabet $\mathcal{Y}$.
        """
        return int(np.amax(self.outputs))

    @cached_property
    def input_edges(self) -> npt.NDArray[np.integer]:
        r"""
        The matrix of input edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the input associated with that edge (an element of $\mathcal{X}$); if there is no such edge, then the element is $-1$.

        Examples:
            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> fsm.input_edges
            array([[ 0,  1, -1, -1],
                   [-1, -1,  0,  1],
                   [ 0,  1, -1, -1],
                   [-1, -1,  0,  1]])
        """
        input_edges = np.full((self.num_states, self.num_states), fill_value=-1)
        for state_from in range(self.num_states):
            for x, state_to in enumerate(self.next_states[state_from, :]):
                input_edges[state_from, state_to] = x
        return input_edges

    @cached_property
    def output_edges(self) -> npt.NDArray[np.integer]:
        r"""
        The matrix of output edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the output associated with that edge (an element of $\mathcal{Y}$); if there is no such edge, then the element is $-1$.

        Examples:
            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> fsm.output_edges
            array([[ 0,  3, -1, -1],
                   [-1, -1,  1,  2],
                   [ 3,  0, -1, -1],
                   [-1, -1,  2,  1]])
        """
        output_edges = np.full((self.num_states, self.num_states), fill_value=-1)
        for state_from in range(self.num_states):
            for x, state_to in enumerate(self.next_states[state_from, :]):
                output_edges[state_from, state_to] = self.outputs[state_from, x]
        return output_edges

    def process(
        self,
        input_sequence: npt.ArrayLike,
        initial_state: int,
    ) -> tuple[npt.NDArray[np.integer], int]:
        r"""
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state $s_\mathrm{i}$. The input sequence and the output sequence are denoted by $\mathbf{x} = (x_0, x_1, \ldots, x_{L-1}) \in \mathcal{X}^L$ and $\mathbf{y} = (y_0, y_1, \ldots, y_{L-1}) \in \mathcal{Y}^L$, respectively.

        Parameters:
            input_sequence: The input sequence $\mathbf{x} \in \mathcal{X}^L$. It should be a 1D-array with elements in $\mathcal{X}$.

            initial_state: The initial state $s_\mathrm{i}$ of the machine. Should be an integer in $\mathcal{S}$.

        Returns:
            output_sequence: The output sequence $\mathbf{y} \in \mathcal{Y}^L$ corresponding to `input_sequence`, assuming the machine starts at the state given by `initial_state`. It is a 1D-array with elements in $\mathcal{Y}$.

            final_state: The final state $s_\mathrm{f}$ of the machine. It is an integer in $\mathcal{S}$.

        Examples:
            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> input_sequence, initial_state = [1, 1, 0, 1, 0], 0
            >>> output_sequence, final_state = fsm.process(input_sequence, initial_state)
            >>> output_sequence
            array([3, 2, 2, 0, 1])
            >>> final_state
            2
        """
        output_sequence = np.empty_like(input_sequence, dtype=int)
        s = initial_state
        for t, x in np.ndenumerate(input_sequence):
            y = self.outputs[s, x]
            s = self.next_states[s, x]
            output_sequence[t] = y
        final_state = int(s)
        return output_sequence, final_state

    def viterbi(
        self,
        observed_sequence: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        initial_metrics: npt.ArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        r"""
        Applies the Viterbi algorithm on a given observed sequence. The Viterbi algorithm finds the most probable input sequence $\hat{\mathbf{x}}(s) \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$, given an observed sequence $\mathbf{z} \in \mathcal{Z}^L$. It is assumed uniform input priors. See <cite>LC04, Sec. 12.1</cite>.

        Parameters:
            observed_sequence: The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            initial_metrics: The initial metrics for each state. It must be a 1D-array of length $|\mathcal{S}|$. The default value is `0.0` for all states.

        Returns:
            input_sequences_hat: The most probable input sequence $\hat{\mathbf{x}}(s) \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$. It is a 2D-array of shape $L \times |\mathcal{S}|$, in which column $s$ is equal to $\hat{\mathbf{x}}(s)$.

            final_metrics: The final metrics for each state. It is a 1D-array of length $|\mathcal{S}|$.
        """
        L, num_states = len(observed_sequence), self.num_states
        choices = np.empty((L, num_states), dtype=int)
        metrics = np.full((L + 1, num_states), fill_value=np.inf)
        if initial_metrics is None:
            metrics[0, :] = np.zeros(num_states, dtype=float)
        else:
            metrics[0, :] = initial_metrics
        for t, z in enumerate(observed_sequence):
            for s0 in range(num_states):
                for s1, y in zip(self.next_states[s0], self.outputs[s0]):
                    candidate_metrics = metrics[t, s0] + metric_function(y, z)
                    if candidate_metrics < metrics[t + 1, s1]:
                        metrics[t + 1, s1] = candidate_metrics
                        choices[t, s1] = s0

        # Backtrack
        input_sequences_hat = np.empty((L, num_states), dtype=int)
        for final_state in range(num_states):
            s1 = final_state
            for t in reversed(range(L)):
                s0 = choices[t, s1]
                input_sequences_hat[t, final_state] = self.input_edges[s0, s1]
                s1 = s0

        return input_sequences_hat, metrics[L, :]

    def viterbi_streaming(
        self,
        observed_sequence: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        memory: MetricMemory,
    ) -> npt.NDArray[np.integer]:
        r"""
        Applies the streaming version of the Viterbi algorithm on a given observed sequence. The path memory (or traceback length) is denoted by $\tau$. It chooses the survivor with best metric and selects the information block on this path. See <cite>LC04, Sec. 12.3</cite>.

        Parameters:
            observed_sequence: The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            memory: The metrics for each state. It must be a dictionary containing two keys: `'paths'`, a 2D-array of integers of shape $|\mathcal{S}| \times (\tau + 1)$; and `'metrics'`, a 2D-array of floats of shape $|\mathcal{S}| \times (\tau + 1)$. This dictionary is updated in-place by this method.

        Returns:
            input_sequence_hat: The most probable input sequence $\hat{\mathbf{x}} \in \mathcal{X}^L$
        """
        num_states = self.num_states
        input_sequences_hat = np.empty(len(observed_sequence), dtype=int)
        for t, z in enumerate(observed_sequence):
            new_metrics = np.full(num_states, fill_value=np.inf)
            choices = np.zeros(num_states, dtype=int)
            for s0 in range(num_states):
                for s1, y in zip(self.next_states[s0], self.outputs[s0]):
                    candidate_metric = memory["metrics"][s0, -1] + metric_function(y, z)
                    if candidate_metric < new_metrics[s1]:
                        new_metrics[s1] = candidate_metric
                        choices[s1] = s0

            s_star = np.argmin(new_metrics)
            s0, s1 = memory["paths"][s_star, :2]
            input_sequences_hat[t] = self.input_edges[s0, s1]

            memory["metrics"] = np.roll(memory["metrics"], shift=-1, axis=1)
            memory["metrics"][:, -1] = new_metrics
            memory["paths"] = np.roll(memory["paths"], shift=-1, axis=1)

            paths_copy = np.copy(memory["paths"])
            for s1, s0 in enumerate(choices):
                memory["paths"][s1, :-1] = paths_copy[s0, :-1]
                memory["paths"][s1, -1] = s1

        return input_sequences_hat

    def forward_backward(
        self,
        observed_sequence: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        input_priors: npt.ArrayLike | None = None,
        initial_state_distribution: npt.ArrayLike | None = None,
        final_state_distribution: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Applies the forward-backward algorithm on a given observed sequence. The forward-backward algorithm computes the posterior pmf of each input $x_0, x_1, \ldots, x_{L-1} \in \mathcal{X}$ given an observed sequence $\mathbf{z} = (z_0, z_1, \ldots, z_{L-1}) \in \mathcal{Z}^L$. The prior pmf of each input may also be provided. See <cite>LC04, 12.6</cite>.

        Parameters:
            observed_sequence: The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            input_priors: The prior pmf of each input, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ should be $p(x_t = x)$. The default value yields uniform priors.

            initial_state_distribution: The pmf of the initial state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

            final_state_distribution: The pmf of the final state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

        Returns:
            input_posteriors: The posterior pmf of each input, given the observed sequence, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ is $p(x_t = x \mid \mathbf{z})$.
        """
        L, num_states, num_input_symbols = (
            len(observed_sequence),
            self.num_states,
            self.num_input_symbols,
        )

        if input_priors is None:
            input_priors = np.ones((L, num_input_symbols)) / num_input_symbols
        if initial_state_distribution is None:
            initial_state_distribution = np.ones(num_states) / num_states
        if final_state_distribution is None:
            final_state_distribution = np.ones(num_states) / num_states

        log_gamma = np.full((L, num_states, num_states), fill_value=-np.inf)
        log_alpha = np.full((L + 1, num_states), fill_value=-np.inf)
        log_beta = np.full((L + 1, num_states), fill_value=-np.inf)

        with np.errstate(divide="ignore"):
            log_input_priors = np.log(input_priors)
            log_alpha[0, :] = np.log(initial_state_distribution)
            log_beta[L, :] = np.log(final_state_distribution)

        for t, z in enumerate(observed_sequence):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self.outputs[s0, x], self.next_states[s0, x]
                log_gamma[t, s0, s1] = log_input_priors[t, x] + metric_function(y, z)

        for t in range(0, L - 1):
            for s1 in range(num_states):
                log_alpha[t + 1, s1] = np.logaddexp.reduce(
                    log_gamma[t, :, s1] + log_alpha[t, :]
                )

        for t in range(L - 1, -1, -1):
            for s0 in range(num_states):
                log_beta[t, s0] = np.logaddexp.reduce(
                    log_gamma[t, s0, :] + log_beta[t + 1, :]
                )

        log_input_posteriors = np.empty((L, num_input_symbols), dtype=float)
        edge_labels = np.empty(num_states, dtype=float)
        for t in range(L):
            for x in range(num_input_symbols):
                for s0 in range(num_states):
                    s1 = self.next_states[s0, x]
                    edge_labels[s0] = (
                        log_alpha[t, s0] + log_gamma[t, s0, s1] + log_beta[t + 1, s1]
                    )
                log_input_posteriors[t, x] = np.logaddexp.reduce(edge_labels)

        input_posteriors = np.exp(
            log_input_posteriors - np.amax(log_input_posteriors, axis=1, keepdims=True)
        )
        input_posteriors /= np.sum(input_posteriors, axis=1, keepdims=True)

        return input_posteriors
