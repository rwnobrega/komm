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
        encoder = self.code.trellis_encoder()
        self._trellis = encoder.trellis
        self._num_input_bits = encoder.num_input_bits
        self._num_output_bits = encoder.num_output_bits
        self._kept = encoder.kept
        self._message_positions = encoder.message_positions
        trellis, n = self._trellis, self._num_output_bits
        self._bits = int_to_bits(range(2**n), width=n).reshape(-1, n)
        # About 64 MiB of decisions and metrics
        section_bytes = int(trellis.widths.max()) + 8 * 2**n
        self._chunk_size = max(1, 2**26 // (len(trellis) * section_bytes))

    def _decode(
        self,
        input: npt.ArrayLike,
        symbols: Literal["input", "output"],
        positions: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.integer]:
        # Bits of the given symbols on the best path
        trellis, n = self._trellis, self._num_output_bits
        width = self._num_input_bits if symbols == "input" else n

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size, "Decoding with Viterbi algorithm")
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
            # Deleted bits as null L-values
            lu = np.zeros((li.shape[0], len(trellis) * n))
            lu[:, self._kept] = li
            branch_metrics = lu.reshape(lu.shape[0], -1, n) @ self._bits.T
            sequence = viterbi(trellis, branch_metrics, symbols=symbols)
            return int_to_bits(sequence, width=width)[:, positions]

        if self.input_type == "hard":
            li = (-1.0) ** validate_binary_array(input)  # Bits as unit L-values
        else:  # self.input_type == "soft"
            li = validate_soft_array(input)
        return decode(li)

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
        """
        return self._decode(input, "output", self._kept)

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
        if self._message_positions is None:
            v_hat = self.decode_to_codeword(input)
            u_hat = self.code.project_word(v_hat)
        else:
            u_hat = self._decode(input, "input", self._message_positions)
        return u_hat
