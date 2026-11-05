from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_decoders.util import branch_costs
from .._finite_state_machine.TrellisSection import TrellisSection
from .._util.bit_operations import int_to_bits
from .._util.validators import (
    validate_binary_array,
    validate_decision_type,
    validate_integer,
    validate_nonnegative_integer,
    validate_soft_array,
)
from ..types import Array1D


@dataclass
class ViterbiStreamDecoder:
    r"""
    Convolutional stream decoder using Viterbi algorithm. Decode a (hard or soft) bit stream given a [convolutional code](/ref/ConvolutionalCode), assuming a traceback length (path memory) of $\tau$. At time $t$, the decoder chooses the survivor with best metric and outputs its information bits from time $t - \tau$. The output stream has a delay equal to $k \tau$, where $k$ is the number of input bits of the convolutional code. As a rule of thumb, the traceback length is chosen as $\tau = 5\mu$, where $\mu$ is the memory order of the convolutional code. For more details, see <cite>LC04, Sec. 12.4</cite>.

    Parameters:
        convolutional_code: The convolutional code.
        traceback_length: The traceback length (path memory) $\tau$ of the decoder. Must be a non-negative integer.
        initial_state: The initial state of the decoder, numbered as in the [finite-state machine](/ref/ConvolutionalCode/#finite_state_machine) of the convolutional code. Must be in $[0 : 2^\sigma)$, where $\sigma$ is the degree of the convolutional code. The default value is `0`.
        input_type: The type of the input sequence, either `'hard'` or `'soft'`. The default value is `'hard'`.
    """

    convolutional_code: abc.ConvolutionalCode
    traceback_length: int
    initial_state: int = 0
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self):
        self.traceback_length = validate_nonnegative_integer(self.traceback_length)
        self.input_type = validate_decision_type(self.input_type)
        self._fsm = self.convolutional_code.finite_state_machine()
        self.initial_state = validate_integer(
            self.initial_state, low=0, high=self._fsm.num_states
        )
        self._section = TrellisSection(self._fsm.transitions, self._fsm.outputs)
        self._reset()

    def _reset(self) -> None:
        num_states, traceback_length = self._fsm.num_states, self.traceback_length
        self._paths = np.zeros((num_states, traceback_length + 1), dtype=int)
        self._metrics = np.full(num_states, fill_value=np.inf)
        self._metrics[self.initial_state] = 0.0

    def _decode(self, branch_metrics: npt.NDArray[np.floating]) -> Array1D[np.integer]:
        in_states, in_inputs, in_outputs = self._section.incoming
        states = np.arange(self._section.num_next_states)
        input_hat = np.empty(branch_metrics.shape[0], dtype=int)
        for t, gamma in enumerate(branch_metrics):
            # Add, compare, select
            candidates = self._metrics[in_states] + gamma[in_outputs]
            d = np.argmin(candidates, axis=1)
            self._metrics = candidates[states, d]
            survivors = self._paths[in_states[states, d], 1:]
            self._paths = np.column_stack([survivors, in_inputs[states, d]])
            s_star = np.argmin(self._metrics)
            input_hat[t] = self._paths[s_star, 0]  # Oldest input on best survivor
        return input_hat

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Parameters:
            input: The (hard or soft) bit sequence to be decoded. Must be a 1D-array whose length is a multiple of $n$, the number of output bits of the convolutional code.

        Returns:
            output: The decoded bit sequence.

        Examples:
            >>> decoder = komm.ViterbiStreamDecoder(
            ...     convolutional_code=komm.ConvolutionalCode([[0b111, 0b101]]),
            ...     traceback_length=10,
            ... )
            >>> decoder.decode([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0])
            >>> decoder.decode([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
            array([0, 0, 0, 0, 0, 0, 0, 0])
            >>> decoder.decode([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            array([0, 0, 1, 0, 1, 1, 1, 0])
        """
        if self.input_type == "hard":
            li = (-1.0) ** validate_binary_array(input, ndim=1)  # Bits as unit L-values
        else:  # self.input_type == "soft"
            li = validate_soft_array(input, ndim=1)
        n = self.convolutional_code.num_output_bits
        k = self.convolutional_code.num_input_bits
        if not li.size % n == 0:
            raise ValueError(
                f"length of 'input' must be a multiple of {n} (got {li.size})"
            )
        input_hat = self._decode(branch_costs(li.reshape(-1, n)))
        output = int_to_bits(input_hat, width=k)
        return output

    def flush(self) -> Array1D[np.integer]:
        r"""
        Returns the last $k \tau$ bits of the stream, taken from the survivor with best metric, and resets the decoder to its initial state.

        Returns:
            output: The last decoded bits.

        Examples:
            >>> decoder = komm.ViterbiStreamDecoder(
            ...     convolutional_code=komm.ConvolutionalCode([[0b111, 0b101]]),
            ...     traceback_length=4,
            ... )
            >>> decoder.decode([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
            array([0, 0, 0, 0, 1, 0])
            >>> decoder.flush()
            array([1, 1, 1, 0])
        """
        k = self.convolutional_code.num_input_bits
        s_star = np.argmin(self._metrics)
        input_hat = self._paths[s_star, 1:]
        self._reset()
        return int_to_bits(input_hat, width=k)
