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


class MealyMachine:
    r"""
    Finite-state Mealy machine. It is defined by a *set of states* $\mathcal{S}$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S}$, and an *output function* $G : \mathcal{S} \times \mathcal{X} \to \mathcal{Y}$. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, $\mathcal{X} = [0 : |\mathcal{X}|)$, and $\mathcal{Y} = [0 : |\mathcal{Y}|)$, respectively. more details, see [Wikipedia: Mealy machine](https://en.wikipedia.org/wiki/Mealy_machine).


    Parameters:
        transitions: The matrix of transitions of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}$, that is, the next state of the machine given that the current state is $s$ and the input is $x$.

        outputs: The matrix of outputs of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $G(s, x) \in \mathcal{Y}$, that is, the output of the machine given that the current state is $s$ and the input is $x$.

    Examples:
        1. Consider the finite-state Mealy machine whose state diagram depicted in the figure below.

            <figure markdown>
            ![Finite-state Mealy machine example.](/fig/mealy.svg)
            </figure>

            It has set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$. The transition function $T$ and output function $G$ are given by the table below.

            | State $s$ | Input $x$ | Transition $T(s, x)$ | Output $G(s, x)$ |
            | :-------: | :-------: | :------------------: | :--------------: |
            | $0$       | $0$       | $0$                  | $0$              |
            | $0$       | $1$       | $1$                  | $3$              |
            | $1$       | $0$       | $2$                  | $1$              |
            | $1$       | $1$       | $3$                  | $2$              |
            | $2$       | $0$       | $0$                  | $3$              |
            | $2$       | $1$       | $1$                  | $0$              |
            | $3$       | $0$       | $2$                  | $2$              |
            | $3$       | $1$       | $3$                  | $1$              |

                >>> machine = komm.MealyMachine(
                ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
                ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
                ... )
    """

    def __init__(self, transitions: npt.ArrayLike, outputs: npt.ArrayLike):
        self.transitions = np.asarray(transitions)
        self.outputs = np.asarray(outputs)

    def __repr__(self) -> str:
        args = ", ".join([
            f"transitions={self.transitions.tolist()}",
            f"outputs={self.outputs.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def num_states(self) -> int:
        r"""
        The number of states of the machine.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> machine.num_states
            4
        """
        return self.transitions.shape[0]

    @cached_property
    def num_input_symbols(self) -> int:
        r"""
        The size (cardinality) of the input alphabet $\mathcal{X}$.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> machine.num_input_symbols
            2
        """
        return self.transitions.shape[1]

    @cached_property
    def num_output_symbols(self) -> int:
        r"""
        The size (cardinality) of the output alphabet $\mathcal{Y}$.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> machine.num_output_symbols
            4
        """
        return int(np.amax(self.outputs)) + 1

    @cached_property
    def input_edges(self) -> npt.NDArray[np.integer]:
        r"""
        The matrix of input edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the input associated with that edge (an element of $\mathcal{X}$); if there is no such edge, then the element is $-1$.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> machine.input_edges
            array([[ 0,  1, -1, -1],
                   [-1, -1,  0,  1],
                   [ 0,  1, -1, -1],
                   [-1, -1,  0,  1]])
        """
        input_edges = np.full((self.num_states, self.num_states), fill_value=-1)
        for state_from in range(self.num_states):
            for x, state_to in enumerate(self.transitions[state_from, :]):
                input_edges[state_from, state_to] = x
        return input_edges

    @cached_property
    def output_edges(self) -> npt.NDArray[np.integer]:
        r"""
        The matrix of output edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the output associated with that edge (an element of $\mathcal{Y}$); if there is no such edge, then the element is $-1$.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> machine.output_edges
            array([[ 0,  3, -1, -1],
                   [-1, -1,  1,  2],
                   [ 3,  0, -1, -1],
                   [-1, -1,  2,  1]])
        """
        output_edges = np.full((self.num_states, self.num_states), fill_value=-1)
        for state_from in range(self.num_states):
            for x, state_to in enumerate(self.transitions[state_from, :]):
                output_edges[state_from, state_to] = self.outputs[state_from, x]
        return output_edges

    def process(
        self,
        input: npt.ArrayLike,
        initial_state: int,
    ) -> tuple[npt.NDArray[np.integer], int]:
        r"""
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state $s_\mathrm{i}$. The input sequence and the output sequence are denoted by $x = (x_0, x_1, \ldots, x_{L-1}) \in \mathcal{X}^L$ and $y = (y_0, y_1, \ldots, y_{L-1}) \in \mathcal{Y}^L$, respectively.

        Parameters:
            input: The input sequence $x \in \mathcal{X}^L$. It should be a 1D-array with elements in $\mathcal{X}$.

            initial_state: The initial state $s_\mathrm{i}$ of the machine. Should be an integer in $\mathcal{S}$.

        Returns:
            output: The output sequence $y \in \mathcal{Y}^L$ corresponding to `input`, assuming the machine starts at the state given by `initial_state`. It is a 1D-array with elements in $\mathcal{Y}$.

            final_state: The final state $s_\mathrm{f}$ of the machine. It is an integer in $\mathcal{S}$.

        Examples:
            >>> machine = komm.MealyMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ... )
            >>> input, initial_state = [1, 1, 0, 1, 0], 0
            >>> output, final_state = machine.process(input, initial_state)
            >>> output
            array([3, 2, 2, 0, 1])
            >>> final_state
            2
        """
        output = np.empty_like(input, dtype=int)
        s = initial_state
        for t, x in np.ndenumerate(input):
            y = self.outputs[s, x]
            s = self.transitions[s, x]
            output[t] = y
        final_state = int(s)
        return output, final_state

    def viterbi(
        self,
        observed: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        initial_metrics: npt.ArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        r"""
        Applies the Viterbi algorithm on a given observed sequence. The Viterbi algorithm finds the most probable input sequence $\hat{x} \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$, given an observed sequence $z \in \mathcal{Z}^L$. It is assumed uniform input priors. See <cite>LC04, Sec. 12.1</cite>.

        Parameters:
            observed: The observed sequence $z \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            initial_metrics: The initial metrics for each state. It must be a 1D-array of length $|\mathcal{S}|$. The default value is `0.0` for all states.

        Returns:
            input_hat: The most probable input sequence $\hat{x} \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$. It is a 2D-array of shape $L \times |\mathcal{S}|$, in which column $s$ is equal to $\hat{x}$.

            final_metrics: The final metrics for each state. It is a 1D-array of length $|\mathcal{S}|$.
        """
        L, num_states = len(observed), self.num_states
        choices = np.zeros((L, num_states), dtype=int)
        metrics = np.full((L + 1, num_states), fill_value=np.inf)
        if initial_metrics is None:
            metrics[0, :] = np.zeros(num_states, dtype=float)
        else:
            metrics[0, :] = initial_metrics
        for t, z in enumerate(observed):
            for s0 in range(num_states):
                for s1, y in zip(self.transitions[s0], self.outputs[s0]):
                    candidate_metrics = metrics[t, s0] + metric_function(y, z)
                    if candidate_metrics < metrics[t + 1, s1]:
                        metrics[t + 1, s1] = candidate_metrics
                        choices[t, s1] = s0

        # Backtrack
        input_hat = np.empty((L, num_states), dtype=int)
        for final_state in range(num_states):
            s1 = final_state
            for t in reversed(range(L)):
                s0 = choices[t, s1]
                input_hat[t, final_state] = self.input_edges[s0, s1]
                s1 = s0

        return input_hat, metrics[L, :]

    def viterbi_streaming(
        self,
        observed: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        memory: MetricMemory,
    ) -> npt.NDArray[np.integer]:
        r"""
        Applies the streaming version of the Viterbi algorithm on a given observed sequence. The path memory (or traceback length) is denoted by $\tau$. It chooses the survivor with best metric and selects the information block on this path. See <cite>LC04, Sec. 12.3</cite>.

        Parameters:
            observed: The observed sequence $z \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            memory: The metrics for each state. It must be a dictionary containing two keys: `'paths'`, a 2D-array of integers of shape $|\mathcal{S}| \times (\tau + 1)$; and `'metrics'`, a 2D-array of floats of shape $|\mathcal{S}| \times (\tau + 1)$. This dictionary is updated in-place by this method.

        Returns:
            input_hat: The most probable input sequence $\hat{x} \in \mathcal{X}^L$
        """
        num_states = self.num_states
        input_hat = np.empty(len(observed), dtype=int)
        for t, z in enumerate(observed):
            new_metrics = np.full(num_states, fill_value=np.inf)
            choices = np.zeros(num_states, dtype=int)
            for s0 in range(num_states):
                for s1, y in zip(self.transitions[s0], self.outputs[s0]):
                    candidate_metric = memory["metrics"][s0, -1] + metric_function(y, z)
                    if candidate_metric < new_metrics[s1]:
                        new_metrics[s1] = candidate_metric
                        choices[s1] = s0

            s_star = np.argmin(new_metrics)
            s0, s1 = memory["paths"][s_star, :2]
            input_hat[t] = self.input_edges[s0, s1]

            memory["metrics"] = np.roll(memory["metrics"], shift=-1, axis=1)
            memory["metrics"][:, -1] = new_metrics
            memory["paths"] = np.roll(memory["paths"], shift=-1, axis=1)

            paths_copy = np.copy(memory["paths"])
            for s1, s0 in enumerate(choices):
                memory["paths"][s1, :-1] = paths_copy[s0, :-1]
                memory["paths"][s1, -1] = s1

        return input_hat

    def forward_backward(
        self,
        observed: Sequence[Z] | npt.NDArray[Any],
        metric_function: MetricFunction[Z],
        input_priors: npt.ArrayLike | None = None,
        initial_state_distribution: npt.ArrayLike | None = None,
        final_state_distribution: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""
        Applies the forward-backward algorithm on a given observed sequence. The forward-backward algorithm computes the posterior pmf of each input $x_0, x_1, \ldots, x_{L-1} \in \mathcal{X}$ given an observed sequence $z = (z_0, z_1, \ldots, z_{L-1}) \in \mathcal{Z}^L$. The prior pmf of each input may also be provided. See <cite>LC04, 12.6</cite>.

        Parameters:
            observed: The observed sequence $z \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            input_priors: The prior pmf of each input, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ should be $p(x_t = x)$. The default value yields uniform priors.

            initial_state_distribution: The pmf of the initial state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

            final_state_distribution: The pmf of the final state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

        Returns:
            input_posteriors: The posterior pmf of each input, given the observed sequence, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ is $p(x_t = x \mid z)$.
        """
        L, num_states, num_input_symbols = (
            len(observed),
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

        for t, z in enumerate(observed):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self.outputs[s0, x], self.transitions[s0, x]
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
                    s1 = self.transitions[s0, x]
                    edge_labels[s0] = (
                        log_alpha[t, s0] + log_gamma[t, s0, s1] + log_beta[t + 1, s1]
                    )
                log_input_posteriors[t, x] = np.logaddexp.reduce(edge_labels)

        input_posteriors = np.exp(
            log_input_posteriors - np.amax(log_input_posteriors, axis=1, keepdims=True)
        )
        input_posteriors /= np.sum(input_posteriors, axis=1, keepdims=True)

        return input_posteriors
