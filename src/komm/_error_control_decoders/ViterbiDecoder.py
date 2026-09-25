from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_convolutional.TerminatedConvolutionalCode import (
    TerminatedConvolutionalCode,
)
from .._finite_state_machine.trellis import TrellisSection, viterbi
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, chunkwise, with_pbar
from .util import get_pbar


@dataclass
class ViterbiDecoder(abc.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Viterbi decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.1</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Notes:
        - Input type: `hard` (bits) or `soft` (L-values).
        - Output type: `hard` (bits).
    """

    code: TerminatedConvolutionalCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError(
                "Viterbi algorithm not implemented for 'tail-biting'"
            )
        if self.input_type not in ["hard", "soft"]:
            raise ValueError("input_type must be 'hard' or 'soft'")
        fsm = self.code.convolutional_code.finite_state_machine()
        n = self.code.convolutional_code.num_output_bits
        num_steps = self.code.length // n
        section = TrellisSection(fsm.transitions, fsm.outputs, fsm.num_states)
        self._sections = [section] * num_steps
        self._bits = int_to_bits(range(2**n), width=n).reshape(-1, n)
        initial, final = self.code.strategy.initial_final_distributions(fsm.num_states)
        with np.errstate(divide="ignore"):
            self._initial_metrics = -np.log(initial)
            self._final_metrics = -np.log(final)
        # About 64 MiB of decisions and metrics
        step_bytes = fsm.num_states + 8 * 2**n
        self._chunk_size = max(1, 2**26 // (num_steps * step_bytes))

    def _branch_metrics(
        self, r: npt.NDArray[np.integer | np.floating]
    ) -> npt.NDArray[np.integer | np.floating]:
        if self.input_type == "hard":
            r = (-1) ** r  # Bits as unit L-values
        return r @ self._bits.T

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b111, 0b101]]),
            ...     num_blocks=4,
            ...     mode="direct-truncation",
            ... )

            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode([1, 1, 1, 1, 1, 0, 0, 0])
            array([1, 0, 0, 0])

            >>> decoder = komm.ViterbiDecoder(code, input_type="soft")
            >>> decoder.decode([-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8])
            array([1, 0, 0, 0])
        """
        k = self.code.convolutional_code.num_input_bits
        n = self.code.convolutional_code.num_output_bits
        h = self.code.num_blocks

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size)
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Viterbi"))
        def decode(r: npt.NDArray[np.integer | np.floating]):
            x_hat = viterbi(
                sections=self._sections,
                branch_metrics=self._branch_metrics(r.reshape(r.shape[0], -1, n)),
                initial_metrics=self._initial_metrics,
                final_metrics=self._final_metrics,
            )
            u_hat = int_to_bits(x_hat[:, :h], width=k)
            return u_hat

        return decode(input)
