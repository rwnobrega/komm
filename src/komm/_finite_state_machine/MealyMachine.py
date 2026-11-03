from collections.abc import Callable
from functools import cached_property
from itertools import product
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer, validate_integer_array
from ..types import Array1D, Array2D


class MetricMemory(TypedDict):
    paths: Array2D[np.integer]
    metrics: Array1D[np.floating]


class MealyMachine:
    r"""
    Finite-state Mealy machine. It is defined by a *set of states* $\mathcal{S}$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S}$, and an *output function* $G : \mathcal{S} \times \mathcal{X} \to \mathcal{Y}$. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, $\mathcal{X} = [0 : |\mathcal{X}|)$, and $\mathcal{Y} = [0 : |\mathcal{Y}|)$, respectively. For more details, see [Wikipedia: Mealy machine](https://en.wikipedia.org/wiki/Mealy_machine).

    Parameters:
        transitions: The matrix of transitions of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}$, that is, the next state of the machine given that the current state is $s$ and the input is $x$.

        outputs: The matrix of outputs of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $G(s, x) \in \mathcal{Y}$, that is, the output of the machine given that the current state is $s$ and the input is $x$.

    Examples:
        1. Consider the finite-state Mealy machine whose state diagram is depicted in the figure below.

            <figure markdown>
            ![Finite-state Mealy machine example.](/fig/machine_mealy.svg)
            </figure>

            It has the set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$. The transition function $T$ and output function $G$ are given by the table below.

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
        self.transitions = validate_integer_array(transitions, ndim=2)
        calS, calX = self.transitions.shape
        validate_integer_array(self.transitions, low=0, high=calS)
        self.outputs = validate_integer_array(outputs, low=0, shape=(calS, calX))

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

    def process(
        self,
        input: npt.ArrayLike,
        initial_state: int,
    ) -> tuple[Array1D[np.integer], int]:
        r"""
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state $s_\mathrm{i}$. The input sequence and the output sequence are denoted by $x = (x_0, x_1, \ldots, x_{L-1}) \in \mathcal{X}^L$ and $y = (y_0, y_1, \ldots, y_{L-1}) \in \mathcal{Y}^L$, respectively.

        Parameters:
            input: The input sequence $x \in \mathcal{X}^L$. Must be a 1D-array with elements in $\mathcal{X}$.

            initial_state: The initial state $s_\mathrm{i}$ of the machine. Must be an integer in $\mathcal{S}$.

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
        calX = self.num_input_symbols
        input = validate_integer_array(input, low=0, high=calX, ndim=1)
        initial_state = validate_integer(initial_state, low=0, high=self.num_states)
        output = np.empty_like(input, dtype=int)
        s = initial_state
        for t, x in np.ndenumerate(input):
            y = self.outputs[s, x]
            s = self.transitions[s, x]
            output[t] = y
        final_state = int(s)
        return output, final_state

    def viterbi_streaming(
        self,
        observed: npt.ArrayLike,
        metric_function: Callable[[int, Any], float],
        memory: MetricMemory,
    ) -> Array1D[np.integer]:
        r"""
        Applies the streaming version of the Viterbi algorithm on a given observed sequence. The path memory (or traceback length) is denoted by $\tau$. It chooses the survivor with best metric and selects the information block on this path. See <cite>LC04, Sec. 12.4</cite>.

        Parameters:
            observed: The observed sequence $z \in \mathcal{Z}^L$.

            metric_function: The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            memory: The metrics for each state. It must be a dictionary containing two keys: `'paths'`, a 2D-array of integers of shape $|\mathcal{S}| \times (\tau + 1)$, with the inputs along the survivor ending in each state; and `'metrics'`, a 1D-array of floats of length $|\mathcal{S}|$, with the metric of the survivor ending in each state. This dictionary is updated in-place by this method.

        Returns:
            input_hat: The most probable input sequence $\hat{x} \in \mathcal{X}^L$.
        """
        observed = np.asarray(observed)
        L, num_states = observed.shape[0], self.num_states
        input_hat = np.empty(L, dtype=int)
        for t, z in enumerate(observed):
            new_metrics = np.full(num_states, fill_value=np.inf)
            choices = np.zeros((num_states, 2), dtype=int)  # State and input
            for s0, x in product(range(num_states), range(self.num_input_symbols)):
                s1, y = self.transitions[s0, x], self.outputs[s0, x]
                candidate_metric = memory["metrics"][s0] + metric_function(y, z)
                if candidate_metric < new_metrics[s1]:
                    new_metrics[s1] = candidate_metric
                    choices[s1] = s0, x

            memory["metrics"] = new_metrics
            memory["paths"] = np.roll(memory["paths"], shift=-1, axis=1)

            paths_copy = np.copy(memory["paths"])
            for s1, (s0, x) in enumerate(choices):
                memory["paths"][s1, :-1] = paths_copy[s0, :-1]
                memory["paths"][s1, -1] = x

            s_star = np.argmin(new_metrics)
            input_hat[t] = memory["paths"][s_star, 0]  # Oldest input on best survivor

        return input_hat
