from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_convolutional import TerminatedConvolutionalCode
from .._util.bit_operations import int_to_bits
from .._util.decorators import vectorized_method


@dataclass
class ViterbiDecoder(abc.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Viterbi decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.1</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        snr: The signal-to-noise ratio (SNR) of the channel (linear, not decibel). Only used for soft-input decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Parameters: Input:
        r: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

    Parameters: Output:
        u_hat: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

    Notes:
        - Input type: `hard` or `soft`.
        - Output type: `hard`.

    Examples:
        >>> convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b011, 0b101, 0b111]])
        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode="zero-termination")
        >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
        >>> decoder([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])
        array([1, 1, 0, 0, 1])

        >>> convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=4, mode="direct-truncation")
        >>> decoder = komm.ViterbiDecoder(code, input_type="soft", snr=10.0)
        >>> decoder([-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8])
        array([1, 0, 0, 0])

    """

    code: TerminatedConvolutionalCode
    snr: float = 1.0
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError("algorithm not implemented for 'tail-biting'")
        if self.input_type == "hard":
            self.metric_function = self._metric_function_hard
        elif self.input_type == "soft":
            self.metric_function = self._metric_function_soft
        else:
            raise ValueError("input_type must be 'hard' or 'soft'")
        self.k = self.code.convolutional_code.num_input_bits
        self.n = self.code.convolutional_code.num_output_bits
        self.mu = self.code.convolutional_code.memory_order
        self.fsm = self.code.convolutional_code.finite_state_machine()
        self.cache_bit = int_to_bits(range(2**self.n), width=self.n)
        self.initial_metrics = np.full(self.fsm.num_states, fill_value=np.inf)
        self.initial_metrics[0] = 0.0

    def _metric_function_hard(self, y: int, z: float) -> float:
        return np.count_nonzero(self.cache_bit[y] != z)

    def _metric_function_soft(self, y: int, z: int) -> float:
        return np.dot(self.cache_bit[y], z)

    @vectorized_method
    def _decode(
        self, r: npt.NDArray[np.float64 | np.integer]
    ) -> npt.NDArray[np.integer]:
        xs_hat, final_metrics = self.fsm.viterbi(
            observed_sequence=r.reshape(-1, self.n),
            metric_function=self.metric_function,
            initial_metrics=self.initial_metrics,
        )

        if self.code.mode == "direct-truncation":
            s_hat = np.argmin(final_metrics)
            x_hat = xs_hat[:, s_hat]
        else:  # code.mode == "zero-termination"
            x_hat = xs_hat[:, 0][: -self.mu]

        u_hat = int_to_bits(x_hat, width=self.k).ravel()
        return u_hat
