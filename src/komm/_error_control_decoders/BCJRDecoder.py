from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_convolutional.TerminatedConvolutionalCode import (
    TerminatedConvolutionalCode,
)
from .._finite_state_machine.trellis import TrellisSection, forward_backward
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, chunkwise, with_pbar
from .util import get_pbar


@dataclass
class BCJRDecoder(abc.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Bahl–Cocke–Jelinek–Raviv (BCJR) decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.6</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        output_type: The type of the output. Either `'hard'` or `'soft'`. Default is `'soft'`.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits) or `soft` (L-values).
    """

    code: TerminatedConvolutionalCode
    output_type: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError(
                "BCJR algorithm not implemented for 'tail-biting'"
            )
        if self.output_type not in ["hard", "soft"]:
            raise ValueError("'output_type' must be 'hard' or 'soft'")
        fsm = self.code.convolutional_code.finite_state_machine()
        n = self.code.convolutional_code.num_output_bits
        k = self.code.convolutional_code.num_input_bits
        num_steps = self.code.length // n
        section = TrellisSection(fsm.transitions, fsm.outputs, fsm.num_states)
        self._sections = [section] * num_steps
        self._polar = (-1) ** int_to_bits(range(2**n), width=n).reshape(-1, n)
        initial, final = self.code.strategy.initial_final_distributions(fsm.num_states)
        with np.errstate(divide="ignore"):
            self._initial_metrics = np.log(initial)
            self._final_metrics = np.log(final)
        # Input symbols are LSB-first
        self._input_bits = int_to_bits(range(2**k), width=k).reshape(-1, k)
        # About 64 MiB of metrics
        step_bytes = 8 * (fsm.num_states + 2**n + 2**k)
        self._chunk_size = max(1, 2**26 // (num_steps * step_bytes))

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            ...     num_blocks=3,
            ...     mode="zero-termination",
            ... )

            >>> decoder = komm.BCJRDecoder(code)
            >>> decoder.decode([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6])
            array([-0.47774884, -0.61545527,  1.03018771])

            >>> decoder = komm.BCJRDecoder(code, output_type="hard")
            >>> decoder.decode([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6])
            array([1, 1, 0])
        """
        n = self.code.convolutional_code.num_output_bits
        h = self.code.num_blocks

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size)
        @with_pbar(get_pbar(np.size(input) // self.code.length, "BCJR"))
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            log_posteriors = forward_backward(
                sections=self._sections,
                branch_metrics=0.5 * li.reshape(li.shape[0], -1, n) @ self._polar.T,
                initial_metrics=self._initial_metrics,
                final_metrics=self._final_metrics,
            )
            lo = _marginalize(log_posteriors[:, :h], self._input_bits)
            return lo.reshape(li.shape[0], -1)

        output = decode(input)
        if self.output_type == "hard":
            output = (output < 0.0).astype(int)
        return output


def _marginalize(
    log_posteriors: npt.NDArray[np.floating], bits: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    # L-value of each bit, in log domain
    metrics = log_posteriors[..., np.newaxis]
    l0 = np.logaddexp.reduce(np.where(bits == 0, metrics, -np.inf), axis=-2)
    l1 = np.logaddexp.reduce(np.where(bits == 1, metrics, -np.inf), axis=-2)
    return l0 - l1
