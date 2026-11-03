from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_convolutional.TerminatedConvolutionalCode import (
    TerminatedConvolutionalCode,
)
from .._finite_state_machine.Trellis import forward_backward
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, chunkwise
from .._util.validators import validate_decision_type, validate_soft_array
from .util import marginalize


@dataclass
class BCJRDecoder(abc.CodewordDecoder[TerminatedConvolutionalCode]):
    r"""
    Bahl–Cocke–Jelinek–Raviv (BCJR) decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). This decoder computes the a posteriori L-value of each message bit or code bit over the trellis of the encoder, assuming equiprobable messages. Its hard decisions, the signs of the L-values, are the bitwise maximum a posteriori (MAP) decisions, and may not form a codeword. For more details, see <cite>LC04, Sec. 12.6</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        output_type: The type of the output. Either `'hard'` or `'soft'`. The default value is `'soft'`.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits) or `soft` (L-values).
    """

    code: TerminatedConvolutionalCode
    output_type: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        self.output_type = validate_decision_type(self.output_type)
        self._encoder = self.code.trellis_encoder()
        trellis = self._encoder.trellis
        k, n = self._encoder.num_input_bits, self._encoder.num_output_bits
        # Symbols are LSB-first
        self._input_bits = int_to_bits(range(2**k), width=k).reshape(-1, k)
        self._output_bits = int_to_bits(range(2**n), width=n).reshape(-1, n)
        self._polar = (-1) ** self._output_bits
        # About 64 MiB of metrics
        section_bytes = 8 * (int(trellis.widths.max()) + 2 * 2**n + 2**k)
        self._chunk_size = max(1, 2**26 // (len(trellis) * section_bytes))

    def _decode(
        self, input: npt.ArrayLike, symbols: Literal["input", "output"]
    ) -> npt.NDArray[np.integer | np.floating]:
        # L-values of the message or code bits
        trellis, n = self._encoder.trellis, self._encoder.num_output_bits
        if symbols == "input":
            bits, positions = self._input_bits, np.arange(self.code.dimension)
        else:  # symbols == "output"
            bits, positions = self._output_bits, self._encoder.kept

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size, "Decoding with BCJR algorithm")
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            # Deleted bits as null L-values
            lu = np.zeros((li.shape[0], len(trellis) * n))
            lu[:, self._encoder.kept] = li
            branch_metrics = 0.5 * lu.reshape(lu.shape[0], -1, n) @ self._polar.T
            log_posteriors = forward_backward(trellis, branch_metrics, symbols=symbols)
            lo = marginalize(log_posteriors, bits)
            return lo.reshape(li.shape[0], -1)[:, positions]

        output = decode(validate_soft_array(input))
        if self.output_type == "hard":
            output = (output < 0.0).astype(int)
        return output

    def decode_to_codeword(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            ...     num_blocks=3,
            ...     mode="zero-termination",
            ... )

            >>> decoder = komm.BCJRDecoder(code)
            >>> decoder.decode_to_codeword(
            ...     [-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6]
            ... )
            array([-0.47774884, -0.47774884, -0.61545527, -0.57490377,  1.03018771,
                   -2.07935845, -2.07935845,         inf])

            >>> decoder = komm.BCJRDecoder(code, output_type="hard")
            >>> decoder.decode_to_codeword(
            ...     [-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6]
            ... )
            array([1, 1, 1, 1, 0, 1, 1, 0])

            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            ...     num_blocks=3,
            ...     mode="zero-termination",
            ...     puncturing_matrix=[[1, 1], [1, 0]],
            ... )
            >>> decoder = komm.BCJRDecoder(code)
            >>> decoder.decode_to_codeword([-0.8, -0.1, -1.0, +1.8, -1.1, -1.6])
            array([-0.30434446, -0.30434446, -0.45888484,  1.45581944, -2.41885407,
                   -2.41885407])
        """
        return self._decode(input, "output")

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

            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            ...     num_blocks=3,
            ...     mode="zero-termination",
            ...     puncturing_matrix=[[1, 1], [1, 0]],
            ... )
            >>> decoder = komm.BCJRDecoder(code)
            >>> decoder.decode([-0.8, -0.1, -1.0, +1.8, -1.1, -1.6])
            array([-0.30434446, -0.45888484,  1.45581944])
        """
        return self._decode(input, "input")
