from collections.abc import Sequence
from functools import cached_property

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.integer]
FloatArray = npt.NDArray[np.floating]


class TrellisSection:
    r"""
    One section of a trellis. It is one step of a time-varying Mealy machine, in which the number of states may change and some branches may be missing. It is defined by a *set of states* $\mathcal{S}$, a *set of next states* $\mathcal{S}'$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, a *set of branches* $\mathcal{B} \subseteq \mathcal{S} \times \mathcal{X}$, a *transition function* $T : \mathcal{B} \to \mathcal{S}'$, and an *output function* $G : \mathcal{B} \to \mathcal{Y}$. The branch $(s, x) \in \mathcal{B}$ leaves state $s$ with input $x$, enters state $T(s, x)$, and has output $G(s, x)$. Here, for simplicity, the sets are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, $\mathcal{S}' = [0 : |\mathcal{S}'|)$, $\mathcal{X} = [0 : |\mathcal{X}|)$, and $\mathcal{Y} = [0 : |\mathcal{Y}|)$.

    A trellis is a sequence of sections, in which the next states of each section are the states of the following one. A [Mealy machine](/ref/MealyMachine) gives the case where all sections are equal, with $\mathcal{S}' = \mathcal{S}$ and $\mathcal{B} = \mathcal{S} \times \mathcal{X}$.

    Parameters:
        transitions: The matrix of transitions of the section, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}'$ if $(s, x) \in \mathcal{B}$, and $-1$ otherwise.

        outputs: The matrix of outputs of the section, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $G(s, x) \in \mathcal{Y}$ if $(s, x) \in \mathcal{B}$, and any element of $\mathcal{Y}$ otherwise.

        num_next_states: The number $|\mathcal{S}'|$ of next states. It cannot be inferred from `transitions`, since a next state may have no incoming branch.
    """

    def __init__(
        self,
        transitions: npt.ArrayLike,
        outputs: npt.ArrayLike,
        num_next_states: int,
    ):
        self.transitions: IntArray = np.asarray(transitions)
        self.outputs: IntArray = np.asarray(outputs)
        self.num_next_states = num_next_states

    @property
    def num_states(self) -> int:
        return self.transitions.shape[0]

    @property
    def num_inputs(self) -> int:
        return self.transitions.shape[1]

    @cached_property
    def incoming(self) -> tuple[IntArray, IntArray, IntArray]:
        r"""
        The branches into each next state, in (state, input) order. They are given by three 2D-arrays, with the state, the input, and the output of each branch; row $s'$ lists the branches into state $s'$. Rows are padded with state $-1$ up to the largest number of branches into a state.
        """
        # Branches in (state, input) order
        states, inputs = np.nonzero(self.transitions >= 0)
        next_states = self.transitions[states, inputs]
        # Group by next state, keeping that order
        order = np.argsort(next_states, kind="stable")
        states, inputs, next_states = states[order], inputs[order], next_states[order]
        counts = np.bincount(next_states, minlength=self.num_next_states)
        # Position of each branch in its group
        starts = np.cumsum(counts) - counts
        slots = np.arange(next_states.size) - np.repeat(starts, counts)
        shape = (self.num_next_states, max(1, int(counts.max(initial=0))))
        in_states = np.full(shape, -1)
        in_inputs = np.zeros(shape, dtype=int)
        in_outputs = np.zeros(shape, dtype=int)
        in_states[next_states, slots] = states
        in_inputs[next_states, slots] = inputs
        in_outputs[next_states, slots] = self.outputs[states, inputs]
        return in_states, in_inputs, in_outputs


def _pad(metrics: FloatArray, value: float) -> FloatArray:
    # Extra column, reached by index -1
    column = np.full((*metrics.shape[:-1], 1), value)
    return np.concatenate([metrics, column], axis=-1)


def viterbi(
    sections: Sequence[TrellisSection],
    branch_metrics: npt.ArrayLike,
    initial_metrics: npt.ArrayLike,
    final_metrics: npt.ArrayLike,
) -> IntArray:
    r"""
    Finds the input sequence of least cost. The cost of a path is the sum of the metrics of its initial state, its branches, and its final state. Ties go to the branch that comes first in (state, input) order.

    Parameters:
        sections: The $L$ sections of the trellis.

        branch_metrics: The cost of each output at each step. Its last two dimensions have lengths $L$ and $|\mathcal{Y}|$. It may have extra leading dimensions, which are kept in the output.

        initial_metrics: The cost of each initial state, along the last dimension; `inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`.

        final_metrics: The cost of each final state, along the last dimension; `inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`.

    Returns:
        inputs: The input sequence of least cost. Has the same shape as `branch_metrics`, but with the last dimension removed.
    """
    branch_metrics = np.asarray(branch_metrics)
    shape = branch_metrics.shape[:-2]
    metrics = np.broadcast_to(initial_metrics, (*shape, sections[0].num_states))
    choices: list[IntArray] = []
    for t, section in enumerate(sections):
        # Add, compare, select
        in_states, _, in_outputs = section.incoming
        padded, gamma = _pad(metrics, np.inf), branch_metrics[..., t, :]
        metrics = padded[..., in_states[:, 0]] + gamma[..., in_outputs[:, 0]]
        choice = np.zeros(metrics.shape, dtype=np.uint8)
        for d in range(1, in_states.shape[1]):
            candidate = padded[..., in_states[:, d]] + gamma[..., in_outputs[:, d]]
            better = candidate < metrics
            metrics[better] = candidate[better]
            choice[better] = d
        choices.append(choice)
    # Trace back from best final state
    final_metrics = np.broadcast_to(final_metrics, metrics.shape)
    states = np.argmin(metrics + final_metrics, axis=-1, keepdims=True)
    inputs = np.empty((*shape, len(sections)), dtype=int)
    for t in reversed(range(len(sections))):
        in_states, in_inputs, _ = sections[t].incoming
        d = np.take_along_axis(choices[t], states, axis=-1)
        inputs[..., t : t + 1] = in_inputs[states, d]
        states = in_states[states, d]
    return inputs
