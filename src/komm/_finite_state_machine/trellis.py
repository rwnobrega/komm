from functools import cached_property

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.integer]


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
