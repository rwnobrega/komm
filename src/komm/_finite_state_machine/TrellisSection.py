from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer_array, validate_positive_integer
from ..types import Array2D


class TrellisSection:
    r"""
    One section of a trellis. It is one step of a time-varying Mealy machine, in which the number of states may change and some branches may be missing. It is defined by a *set of states* $\mathcal{S}$, a *set of next states* $\mathcal{S}'$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, a *set of branches* $\mathcal{B} \subseteq \mathcal{S} \times \mathcal{X}$, a *transition function* $T : \mathcal{B} \to \mathcal{S}'$, and an *output function* $G : \mathcal{B} \to \mathcal{Y}$. The branch $(s, x) \in \mathcal{B}$ leaves state $s$ with input $x$, enters state $T(s, x)$, and has output $G(s, x)$. Here, for simplicity, the sets are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, $\mathcal{S}' = [0 : |\mathcal{S}'|)$, $\mathcal{X} = [0 : |\mathcal{X}|)$, and $\mathcal{Y} = [0 : |\mathcal{Y}|)$.

    A [trellis](/ref/Trellis) is a sequence of sections, in which the next states of each section are the states of the following one. A [Mealy machine](/ref/MealyMachine) gives the case where all sections are equal, with $\mathcal{S}' = \mathcal{S}$ and $\mathcal{B} = \mathcal{S} \times \mathcal{X}$.

    Parameters:
        transitions: The matrix of transitions of the section, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}'$ if $(s, x) \in \mathcal{B}$, and $-1$ otherwise.

        outputs: The matrix of outputs of the section, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $G(s, x) \in \mathcal{Y}$ if $(s, x) \in \mathcal{B}$, and any element of $\mathcal{Y}$ otherwise.

        num_next_states: The number $|\mathcal{S}'|$ of next states. It cannot be inferred from `transitions`, since a next state may have no incoming branch.

    Examples:
        1. Consider the machine of the [Mealy machine](/ref/MealyMachine) example, started at state $0$. Its second section, depicted in the figure below, goes from the states $0$ and $1$, the only ones reachable in one step, to all four states. No branch is missing.

            <figure markdown>
            ![Trellis section from 2 to 4 states.](/fig/trellis_section_2_4.svg)
            </figure>

            It has the set of states $\mathcal{S} = \\{ 0, 1 \\}$, set of next states $\mathcal{S}' = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$, and set of branches $\mathcal{B} = \mathcal{S} \times \mathcal{X}$. The transition function $T$ and output function $G$ are given by the table below.

            | State $s$ | Input $x$ | Transition $T(s, x)$ | Output $G(s, x)$ |
            | :-------: | :-------: | :------------------: | :--------------: |
            | $0$       | $0$       | $0$                  | $0$              |
            | $0$       | $1$       | $1$                  | $3$              |
            | $1$       | $0$       | $2$                  | $1$              |
            | $1$       | $1$       | $3$                  | $2$              |

                >>> section = komm.TrellisSection(
                ...     transitions=[[0, 1], [2, 3]],
                ...     outputs=[[0, 3], [1, 2]],
                ...     num_next_states=4,
                ... )

        2. Now let the same machine return to state $0$ with two zero inputs. The first of these two sections is depicted in the figure below. Only the branches with input $0$ remain; they enter the states $0$ and $2$ of the machine, which are numbered $0$ and $1$ in the section.

            <figure markdown>
            ![Trellis section from 4 to 2 states.](/fig/trellis_section_4_2.svg)
            </figure>

            It has the set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, set of next states $\mathcal{S}' = \\{ 0, 1 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$, and set of branches $\mathcal{B} = \mathcal{S} \times \\{ 0 \\}$. The transition function $T$ and output function $G$ are given by the table below. The entries of `outputs` for the missing branches are arbitrary; here, they are those of the machine.

            | State $s$ | Input $x$ | Transition $T(s, x)$ | Output $G(s, x)$ |
            | :-------: | :-------: | :------------------: | :--------------: |
            | $0$       | $0$       | $0$                  | $0$              |
            | $1$       | $0$       | $1$                  | $1$              |
            | $2$       | $0$       | $0$                  | $3$              |
            | $3$       | $0$       | $1$                  | $2$              |

                >>> section = komm.TrellisSection(
                ...     transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
                ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
                ...     num_next_states=2,
                ... )
    """

    def __init__(
        self,
        transitions: npt.ArrayLike,
        outputs: npt.ArrayLike,
        num_next_states: int,
    ):
        self.num_next_states = validate_positive_integer(num_next_states)
        self.transitions: Array2D[np.integer] = validate_integer_array(
            transitions, low=-1, high=self.num_next_states, ndim=2
        )
        self.outputs: Array2D[np.integer] = validate_integer_array(
            outputs, low=0, shape=self.transitions.shape
        )

    def __repr__(self) -> str:
        args = ", ".join([
            f"transitions={self.transitions.tolist()}",
            f"outputs={self.outputs.tolist()}",
            f"num_next_states={self.num_next_states}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    def num_states(self) -> int:
        r"""
        The number of states of the section.

        Examples:
            >>> section = komm.TrellisSection(
            ...     transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ...     num_next_states=2,
            ... )
            >>> section.num_states
            4
        """
        return self.transitions.shape[0]

    @property
    def num_input_symbols(self) -> int:
        r"""
        The size (cardinality) of the input alphabet $\mathcal{X}$.

        Examples:
            >>> section = komm.TrellisSection(
            ...     transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ...     num_next_states=2,
            ... )
            >>> section.num_input_symbols
            2
        """
        return self.transitions.shape[1]

    @cached_property
    def incoming(
        self,
    ) -> tuple[Array2D[np.integer], Array2D[np.integer], Array2D[np.integer]]:
        r"""
        The branches into each next state, in (state, input) order. They are given by three 2D-arrays, with the state, the input, and the output of each branch; row $s'$ lists the branches into state $s'$. Rows are padded with state $-1$ up to the largest number of branches into a state.

        Examples:
            >>> section = komm.TrellisSection(
            ...     transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
            ...     outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
            ...     num_next_states=2,
            ... )
            >>> states, inputs, outputs = section.incoming
            >>> states
            array([[0, 2],
                   [1, 3]])
            >>> inputs
            array([[0, 0],
                   [0, 0]])
            >>> outputs
            array([[0, 3],
                   [1, 2]])
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
