from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._finite_state_machine.Trellis import viterbi
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, chunkwise
from .._util.validators import (
    validate_binary_array,
    validate_decision_type,
    validate_soft_array,
)


@dataclass
class ViterbiDecoder(abc.CodewordDecoder[abc.BlockCode]):
    r"""
    Viterbi decoder for general [block codes](/ref/BlockCode). This decoder searches a trellis of the code for the maximum-likelihood (ML) codeword, which is also the maximum a posteriori (MAP) codeword when messages are equiprobable. For more details, see <cite>LC04, Sec. 12.1</cite> and <cite>MacK03, Ch. 25</cite>.

    [Terminated convolutional codes](/ref/TerminatedConvolutionalCode) are decoded on the trellis of their encoder, whose inputs are the message bits; other codes, on their [minimal trellis](/ref/BlockCode#minimal_trellis), with the message recovered from the decoded codeword.

    Parameters:
        code: The block code to be used for decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. The default value is `'hard'`.

    Notes:
        - Input type: `hard` (bits) or `soft` (L-values).
        - Output type: `hard` (bits).
    """

    code: abc.BlockCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        self.input_type = validate_decision_type(self.input_type)
        self._encoder = self.code.trellis_encoder()
        if self._encoder is None:
            # Minimal trellis: inputs are code bits
            self._trellis = self.code.minimal_trellis()
            self._num_input_bits = 1
            self._num_output_bits = 1
            self._num_sections = self.code.length
            self._kept = np.arange(self.code.length)
        else:
            self._trellis = self._encoder.trellis
            self._num_input_bits = self._encoder.num_input_bits
            self._num_output_bits = self._encoder.num_output_bits
            self._num_sections = self.code.dimension // self._num_input_bits
            self._kept = self._encoder.kept
        trellis, n = self._trellis, self._num_output_bits
        self._bits = int_to_bits(range(2**n), width=n).reshape(-1, n)
        # About 64 MiB of decisions and metrics
        section_bytes = int(trellis.widths.max()) + 8 * 2**n
        self._chunk_size = max(1, 2**26 // (len(trellis) * section_bytes))

    def _best_inputs(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        # Inputs of the best path, as bits
        trellis, h = self._trellis, self._num_sections
        k, n = self._num_input_bits, self._num_output_bits

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size, "Decoding with Viterbi algorithm")
        def best_inputs(li: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
            # Deleted bits as null L-values
            lu = np.zeros((li.shape[0], len(trellis) * n))
            lu[:, self._kept] = li
            branch_metrics = lu.reshape(lu.shape[0], -1, n) @ self._bits.T
            x_hat = viterbi(trellis, branch_metrics)
            return int_to_bits(x_hat[:, :h], width=k)

        if self.input_type == "hard":
            li = (-1.0) ** validate_binary_array(input)  # Bits as unit L-values
        else:  # self.input_type == "soft"
            li = validate_soft_array(input)
        return best_inputs(li)

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b111, 0b101]]),
            ...     num_blocks=4,
            ...     mode="direct-truncation",
            ... )

            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode_to_codeword([1, 1, 1, 1, 1, 0, 0, 0])
            array([1, 1, 1, 0, 1, 1, 0, 0])

            >>> decoder = komm.ViterbiDecoder(code, input_type="soft")
            >>> decoder.decode_to_codeword(
            ...     [-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8]
            ... )
            array([1, 1, 1, 0, 1, 1, 0, 0])

            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b101, 0b111]]),
            ...     num_blocks=4,
            ...     mode="zero-termination",
            ...     puncturing_matrix=[[1, 0], [1, 1]],
            ... )
            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode_to_codeword([1, 1, 0, 1, 1, 0, 0, 0, 0])
            array([1, 1, 1, 1, 1, 0, 0, 0, 0])

            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode_to_codeword([1, 0, 1, 1, 0, 0, 0])
            array([1, 0, 1, 1, 0, 1, 0])

            >>> decoder = komm.ViterbiDecoder(code, input_type="soft")
            >>> decoder.decode_to_codeword([-0.8, +0.9, -1.1, -0.7, +0.4, +0.3, +1.0])
            array([1, 0, 1, 1, 0, 1, 0])
        """
        if self._encoder is None:
            v_hat = self._best_inputs(input)
        else:
            u_hat = self._best_inputs(input)
            v_hat = self.code.encode(u_hat)
        return v_hat

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

            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b101, 0b111]]),
            ...     num_blocks=4,
            ...     mode="zero-termination",
            ...     puncturing_matrix=[[1, 0], [1, 1]],
            ... )
            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode([1, 1, 0, 1, 1, 0, 0, 0, 0])
            array([1, 0, 0, 0])

            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ViterbiDecoder(code, input_type="hard")
            >>> decoder.decode([1, 0, 1, 1, 0, 0, 0])
            array([1, 0, 1, 1])

            >>> decoder = komm.ViterbiDecoder(code, input_type="soft")
            >>> decoder.decode([-0.8, +0.9, -1.1, -0.7, +0.4, +0.3, +1.0])
            array([1, 0, 1, 1])
        """
        if self._encoder is None:
            v_hat = self._best_inputs(input)
            u_hat = self.code.project_word(v_hat)
        else:
            u_hat = self._best_inputs(input)
        return u_hat
