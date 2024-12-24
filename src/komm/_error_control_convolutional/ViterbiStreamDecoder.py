from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._finite_state_machine.FiniteStateMachine import MetricMemory
from .._util.bit_operations import int_to_bits
from .ConvolutionalCode import ConvolutionalCode


@dataclass
class ViterbiStreamDecoder:
    r"""
    Convolutional stream decoder using Viterbi algorithm. Decode a (hard or soft) bit stream given a [convolutional code](/ref/ConvolutionalCode), assuming a traceback length (path memory) of $\tau$. At time $t$, the decoder chooses the path survivor with best metric at time $t - \tau$ and outputs the corresponding information bits. The output stream has a delay equal to $k \tau$, where $k$ is the number of input bits of the convolutional code. As a rule of thumb, the traceback length is chosen as $\tau = 5\mu$, where $\mu$ is the memory order of the convolutional code.

    Attributes:
        convolutional_code: The convolutional code.
        traceback_length: The traceback length (path memory) $\tau$ of the decoder.
        state: The current state of the decoder. The default value is `0`.
        input_type: The type of the input sequence, either `hard` or `soft`. The default value is `hard`.
    """

    convolutional_code: ConvolutionalCode
    traceback_length: int
    state: int = 0
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self):
        fsm = self.convolutional_code.finite_state_machine()
        num_states, traceback_length = fsm.num_states, self.traceback_length
        self.memory: MetricMemory = {
            "paths": np.zeros((num_states, traceback_length + 1), dtype=int),
            "metrics": np.full((num_states, traceback_length + 1), fill_value=np.inf),
        }
        self.memory["metrics"][self.state, -1] = 0.0

    @cached_property
    def cache_bit(self) -> npt.NDArray[np.integer]:
        n = self.convolutional_code.num_output_bits
        return np.array([int_to_bits(y, width=n) for y in range(2**n)])

    def metric_function(self, y: int, z: npt.ArrayLike) -> float:
        if self.input_type == "hard":
            return np.count_nonzero(self.cache_bit[y] != z)
        else:  # self.input_type == "soft"
            return np.dot(self.cache_bit[y], z)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Parameters:
            input: The (hard or soft) bit sequence to be decoded.

        Returns:
            output: The decoded bit sequence.

        Examples:
                >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
                >>> decoder = komm.ViterbiStreamDecoder(convolutional_code, traceback_length=10)
                >>> decoder([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
                array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                >>> decoder(np.zeros(2*10, dtype=int))
                array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        """
        input = np.asarray(input)
        n = self.convolutional_code.num_output_bits
        k = self.convolutional_code.num_input_bits
        fsm = self.convolutional_code.finite_state_machine()
        input_sequence_hat = fsm.viterbi_streaming(
            observed_sequence=input.reshape(-1, n),
            metric_function=self.metric_function,
            memory=self.memory,
        )
        output = int_to_bits(input_sequence_hat, width=k).ravel()
        return output
