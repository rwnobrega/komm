from functools import cached_property

import numpy as np
import numpy.typing as npt


class MooreMachine:
    r"""
    Finite-state Moore machine. It is defined by a *set of states* $\mathcal{S}$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S}$, and an *output function* $G : \mathcal{S} \to \mathcal{Y}$. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$, $\mathcal{X} = [0 : |\mathcal{X}|)$, and $\mathcal{Y} = [0 : |\mathcal{Y}|)$, respectively. more details, see [Wikipedia: Moore machine](https://en.wikipedia.org/wiki/Moore_machine).

    Parameters:
        transitions: The matrix of transitions of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}$, that is, the next state of the machine given that the current state is $s$ and the input is $x$.

        outputs: The vector of outputs of the machine, of shape $|\mathcal{S}|$. The element in position $s \in \mathcal{S}$ should be $G(s) \in \mathcal{Y}$, that is, the output of the machine given that the current state is $s$.

    Examples:
        1. Consider the finite-state Moore machine whose state diagram depicted in the figure below.

            <figure markdown>
            ![Finite-state Moore machine example.](/fig/moore.svg)
            </figure>

            It has set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1 \\}$. The transition function $T$ and output function $G$ are given by the tables below.

            <div>
            <span>

            | State $s$ | Input $x$ | Transition $T(s, x)$ |
            | :-------: | :-------: | :------------------: |
            | $0$       | $0$       | $0$                  |
            | $0$       | $1$       | $1$                  |
            | $1$       | $0$       | $2$                  |
            | $1$       | $1$       | $3$                  |
            | $2$       | $0$       | $0$                  |
            | $2$       | $1$       | $1$                  |
            | $3$       | $0$       | $2$                  |
            | $3$       | $1$       | $3$                  |

            </span>
            <span>

            | State $s$ |  Output $G(s)$ |
            | :-------: |  :-----------: |
            | $0$       |  $0$           |
            | $1$       |  $0$           |
            | $2$       |  $1$           |
            | $3$       |  $1$           |

            </span>
            </div>

                >>> machine = komm.MooreMachine(
                ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
                ...     outputs=[0, 0, 1, 1],
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
            >>> machine = komm.MooreMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[0, 0, 1, 1],
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
            >>> machine = komm.MooreMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[0, 0, 1, 1],
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
            >>> machine = komm.MooreMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[0, 0, 1, 1],
            ... )
            >>> machine.num_output_symbols
            2
        """
        return int(np.amax(self.outputs)) + 1

    def process(
        self,
        input: npt.ArrayLike,
        initial_state: int,
    ) -> tuple[npt.NDArray[np.integer], int]:
        r"""
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state $s_\mathrm{i}$. The input sequence and the output sequence are denoted by $x = (x_0, x_1, \ldots, x_{L-1}) \in \mathcal{X}^L$ and $y = (y_0, y_1, \ldots, y_{L-1}) \in \mathcal{Y}^{L}$, respectively.

        Parameters:
            input: The input sequence $x \in \mathcal{X}^L$. It should be a 1D-array with elements in $\mathcal{X}$.

            initial_state: The initial state $s_\mathrm{i}$ of the machine. Should be an integer in $\mathcal{S}$.

        Returns:
            output: The output sequence $y \in \mathcal{Y}^{L}$ corresponding to `input`, assuming the machine starts at the state given by `initial_state`. It is a 1D-array with elements in $\mathcal{Y}$.

            final_state: The final state $s_\mathrm{f}$ of the machine. It is an integer in $\mathcal{S}$.

        Examples:
            >>> machine = komm.MooreMachine(
            ...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
            ...     outputs=[0, 0, 1, 1],
            ... )
            >>> input, initial_state = [1, 1, 0, 1, 0], 0
            >>> output, final_state = machine.process(input, initial_state)
            >>> output
            array([0, 1, 1, 0, 1])
            >>> final_state
            2
        """
        output = np.empty_like(input, dtype=int)
        s = initial_state
        for t, x in np.ndenumerate(input):
            s = self.transitions[s, x]
            output[t] = self.outputs[s]
        final_state = int(s)
        return output, final_state
