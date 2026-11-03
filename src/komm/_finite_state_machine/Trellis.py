from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_choice
from ..types import Array1D
from .TrellisSection import TrellisSection


class Trellis(Sequence[TrellisSection]):
    r"""
    Trellis. It is a graph whose nodes, called *states*, are grouped into *times* $t \in [0 : L]$, and whose edges, called *branches*, connect states at consecutive times <cite>MacK03, Sec. 25.2</cite>. Here, it is given by a sequence of $L$ [trellis sections](/ref/TrellisSection), where section $t$ holds the branches from time $t$ to time $t + 1$. The next states of each section are the states of the following one.

    Parameters:
        sections: The sections of the trellis. Must be a non-empty sequence, in which the number of next states of each section is equal to the number of states of the following one.

    Examples:
        1. Consider the machine of the [Mealy machine](/ref/MealyMachine) example, started at state $0$. It receives two arbitrary inputs, followed by two zero inputs that return it to state $0$. Its trellis, depicted in the figure below, has four sections; the second and the third are those of the [trellis section](/ref/TrellisSection) examples.

            <figure markdown>
            ![Trellis example.](/fig/trellis.svg)
            </figure>

                >>> trellis = komm.Trellis([
                ...     komm.TrellisSection([[0, 1]],
                ...                         [[0, 3]]),
                ...     komm.TrellisSection([[0, 1], [2, 3]],
                ...                         [[0, 3], [1, 2]]),
                ...     komm.TrellisSection([[0, -1], [1, -1], [0, -1], [1, -1]],
                ...                         [[0, 3], [1, 2], [3, 0], [2, 1]]),
                ...     komm.TrellisSection([[0, -1], [0, -1]],
                ...                         [[0, 3], [3, 0]]),
                ... ])
    """

    def __init__(self, sections: Sequence[TrellisSection]):
        self.sections = tuple(sections)
        if not self.sections:
            raise ValueError("'sections' must be non-empty")
        for t in range(1, len(self.sections)):
            expected = self.sections[t - 1].num_next_states
            actual = self.sections[t].num_states
            if not actual == expected:
                raise ValueError(
                    f"section {t} must have {expected} states (got {actual})"
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sections={list(self.sections)})"

    def __len__(self) -> int:
        return len(self.sections)

    @overload
    def __getitem__(self, index: int) -> TrellisSection: ...

    @overload
    def __getitem__(self, index: slice) -> "Trellis": ...

    def __getitem__(self, index: int | slice) -> "TrellisSection | Trellis":
        if isinstance(index, slice):
            return Trellis(self.sections[index])
        return self.sections[index]

    @property
    def widths(self) -> Array1D[np.integer]:
        r"""
        The widths of the trellis. The width at a given time is its number of states <cite>MacK03, Sec. 25.2</cite>. The sequence of widths, from time $0$ to time $L$, is also called the *state space complexity profile* <cite>LC04, Sec. 9.2</cite>.

        Examples:
            >>> trellis = komm.Trellis([
            ...     komm.TrellisSection([[0, 1]],
            ...                         [[0, 3]]),
            ...     komm.TrellisSection([[0, 1], [2, 3]],
            ...                         [[0, 3], [1, 2]]),
            ...     komm.TrellisSection([[0, -1], [1, -1], [0, -1], [1, -1]],
            ...                         [[0, 3], [1, 2], [3, 0], [2, 1]]),
            ...     komm.TrellisSection([[0, -1], [0, -1]],
            ...                         [[0, 3], [3, 0]]),
            ... ])
            >>> trellis.widths
            array([1, 2, 4, 2, 1])
        """
        widths = [section.num_states for section in self.sections]
        return np.array(widths + [self.sections[-1].num_next_states])

    def viterbi(
        self,
        branch_metrics: npt.ArrayLike,
        initial_metrics: npt.ArrayLike = 0.0,
        final_metrics: npt.ArrayLike = 0.0,
        symbols: Literal["input", "output"] = "input",
    ) -> npt.NDArray[np.integer]:
        r"""
        Finds the input or output sequence of the path of least cost. The cost of a path is the sum of the metrics of its initial state, its branches, and its final state. Ties go to the branch that comes first in (state, input) order.

        Parameters:
            branch_metrics: The cost of each output at each step. Its last two dimensions have lengths $L$ and $|\mathcal{Y}|$. It may have extra leading dimensions, which are kept in the output.

            initial_metrics: The cost of each initial state, along the last dimension; `inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`. The default value is `0.0` for all states.

            final_metrics: The cost of each final state, along the last dimension; `inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`. The default value is `0.0` for all states.

            symbols: The symbols to be returned. Either `'input'` or `'output'`. The default value is `'input'`.

        Returns:
            sequence: The input or output sequence of the path of least cost. Has the same shape as `branch_metrics`, but with the last dimension removed.

        Examples:
            >>> trellis = komm.Trellis([
            ...     komm.TrellisSection([[0, 1]],
            ...                         [[0, 3]]),
            ...     komm.TrellisSection([[0, 1], [2, 3]],
            ...                         [[0, 3], [1, 2]]),
            ...     komm.TrellisSection([[0, -1], [1, -1], [0, -1], [1, -1]],
            ...                         [[0, 3], [1, 2], [3, 0], [2, 1]]),
            ...     komm.TrellisSection([[0, -1], [0, -1]],
            ...                         [[0, 3], [3, 0]]),
            ... ])
            >>> distances = [
            ...     [2, 1, 1, 0],  # Received 11
            ...     [0, 1, 1, 2],  # Received 00
            ...     [2, 1, 1, 0],  # Received 11
            ...     [0, 1, 1, 2],  # Received 00
            ... ]
            >>> trellis.viterbi(distances)
            array([1, 0, 0, 0])
            >>> trellis.viterbi(distances, symbols="output")
            array([3, 1, 3, 0])
        """
        symbols = validate_choice(symbols, ("input", "output"))
        branch_metrics = np.asarray(branch_metrics)
        shape = branch_metrics.shape[:-2]
        metrics = np.broadcast_to(initial_metrics, (*shape, self[0].num_states))
        choices: list[npt.NDArray[np.integer]] = []
        for t, section in enumerate(self):
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
        sequence = np.empty((*shape, len(self)), dtype=int)
        for t in reversed(range(len(self))):
            in_states, in_inputs, in_outputs = self[t].incoming
            in_symbols = in_inputs if symbols == "input" else in_outputs
            d = np.take_along_axis(choices[t], states, axis=-1)
            sequence[..., t : t + 1] = in_symbols[states, d]
            states = in_states[states, d]
        return sequence

    def forward_backward(
        self,
        branch_metrics: npt.ArrayLike,
        initial_metrics: npt.ArrayLike = 0.0,
        final_metrics: npt.ArrayLike = 0.0,
        symbols: Literal["input", "output"] = "input",
    ) -> npt.NDArray[np.floating]:
        r"""
        Computes the a posteriori log-probabilities of the inputs or of the outputs. The weight of a path is the sum of the metrics of its initial state, its branches, and its final state, in the log domain. For the inputs, all sections must have the same number of input symbols.

        Parameters:
            branch_metrics: The log-weight of each output at each step. Its last two dimensions have lengths $L$ and $|\mathcal{Y}|$. It may have extra leading dimensions, which are kept in the output.

            initial_metrics: The log-weight of each initial state, along the last dimension; `-inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`. The default value is `0.0` for all states.

            final_metrics: The log-weight of each final state, along the last dimension; `-inf` forbids a state. The other dimensions are broadcast to the leading dimensions of `branch_metrics`. The default value is `0.0` for all states.

            symbols: The symbols whose posteriors are computed. Either `'input'` or `'output'`. The default value is `'input'`.

        Returns:
            log_posteriors: The a posteriori log-probability of each symbol at each step. Has the same shape as `branch_metrics`, but with the last dimension replaced by the number of input or output symbols.

        Examples:
            >>> trellis = komm.Trellis([
            ...     komm.TrellisSection([[0, 1]],
            ...                         [[0, 3]]),
            ...     komm.TrellisSection([[0, 1], [2, 3]],
            ...                         [[0, 3], [1, 2]]),
            ...     komm.TrellisSection([[0, -1], [1, -1], [0, -1], [1, -1]],
            ...                         [[0, 3], [1, 2], [3, 0], [2, 1]]),
            ...     komm.TrellisSection([[0, -1], [0, -1]],
            ...                         [[0, 3], [3, 0]]),
            ... ])
            >>> distances = [
            ...     [2, 1, 1, 0],  # Received 11
            ...     [0, 1, 1, 2],  # Received 00
            ...     [2, 1, 1, 0],  # Received 11
            ...     [0, 1, 1, 2],  # Received 00
            ... ]
            >>> log_posteriors = trellis.forward_backward(-np.array(distances))
            >>> np.exp(log_posteriors)
            array([[0.04742587, 0.95257413],
                   [0.95257413, 0.04742587],
                   [1.        , 0.        ],
                   [1.        , 0.        ]])
        """
        symbols = validate_choice(symbols, ("input", "output"))
        branch_metrics = np.asarray(branch_metrics)
        shape = branch_metrics.shape[:-2]
        # Forward recursion
        alpha = np.broadcast_to(initial_metrics, (*shape, self[0].num_states))
        alphas = [alpha]
        for t, section in enumerate(self):
            in_states, _, in_outputs = section.incoming
            gamma = branch_metrics[..., t, in_outputs]
            weights = gamma + _pad(alpha, -np.inf)[..., in_states]
            alpha = np.logaddexp.reduce(weights, axis=-1)
            alphas.append(alpha)
        # Backward recursion, with the posteriors
        beta = np.broadcast_to(final_metrics, (*shape, self[-1].num_next_states))
        if symbols == "input":
            log_posteriors = np.empty((*shape, len(self), self[0].num_input_symbols))
        else:  # symbols == "output"
            log_posteriors = np.full(branch_metrics.shape, -np.inf)
        for t in reversed(range(len(self))):
            section = self[t]
            gamma = branch_metrics[..., t, section.outputs]
            weights = gamma + _pad(beta, -np.inf)[..., section.transitions]
            joint = alphas[t][..., np.newaxis] + weights
            if symbols == "input":
                log_posteriors[..., t, :] = np.logaddexp.reduce(joint, axis=-2)
            else:  # symbols == "output"
                for y in np.unique(section.outputs):
                    branches = joint[..., section.outputs == y]
                    log_posteriors[..., t, y] = np.logaddexp.reduce(branches, axis=-1)
            beta = np.logaddexp.reduce(weights, axis=-1)
        normalizer = np.logaddexp.reduce(log_posteriors, axis=-1, keepdims=True)
        return log_posteriors - normalizer


def _pad(metrics: npt.NDArray[np.floating], value: float) -> npt.NDArray[np.floating]:
    # Extra column, reached by index -1
    column = np.full((*metrics.shape[:-1], 1), value)
    return np.concatenate([metrics, column], axis=-1)
