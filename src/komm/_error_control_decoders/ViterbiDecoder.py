from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._error_control_convolutional import TerminatedConvolutionalCode
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, vectorize, with_pbar
from . import base
from .util import get_pbar


@dataclass
class ViterbiDecoder(base.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Viterbi decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.1</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        snr: The signal-to-noise ratio (SNR) of the channel (linear, not decibel). Only used for soft-input decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Notes:
        - Input type: `hard` or `soft`.
        - Output type: `hard`.
    """

    code: TerminatedConvolutionalCode
    snr: float = 1.0
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError(
                "Viterbi algorithm not implemented for 'tail-biting'"
            )
        if self.input_type == "hard":
            self._metric_function = self._metric_function_hard
        elif self.input_type == "soft":
            self._metric_function = self._metric_function_soft
        else:
            raise ValueError("input_type must be 'hard' or 'soft'")
        self._fsm = self.code.convolutional_code.finite_state_machine()
        self._post_process_output = self.code.strategy.viterbi_post_process_output
        n = self.code.convolutional_code.num_output_bits
        self._cache_bit = int_to_bits(range(2**n), width=n)
        self._initial_metrics = np.full(self._fsm.num_states, fill_value=np.inf)
        self._initial_metrics[0] = 0.0

    def _metric_function_hard(self, y: int, z: float) -> float:
        return np.count_nonzero(self._cache_bit[y] != z)

    def _metric_function_soft(self, y: int, z: int) -> float:
        return np.dot(self._cache_bit[y], z)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
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
        k = self.code.convolutional_code.num_input_bits
        n = self.code.convolutional_code.num_output_bits

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Viterbi"))
        def decode(r: npt.NDArray[np.integer]):
            xs_hat, final_metrics = self._fsm.viterbi(
                observed_sequence=r.reshape(-1, n),
                metric_function=self._metric_function,
                initial_metrics=self._initial_metrics,
            )
            x_hat = self._post_process_output(xs_hat, final_metrics)
            u_hat = int_to_bits(x_hat, width=k).ravel()
            return u_hat

        return decode(input)
