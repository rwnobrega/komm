from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._util.bit_operations import bits_to_int, int_to_bits
from .ConvolutionalCode import ConvolutionalCode


@dataclass
class ConvolutionalStreamEncoder:
    r"""
    Convolutional stream encoder. Encode a bit stream using a given [convolutional code](/ref/ConvolutionalCode). The internal state of the encoder is maintained across each call.

    Attributes:
        convolutional_code: The convolutional code.
        state: The current state of the encoder. The default value is `0`.
    """

    convolutional_code: ConvolutionalCode
    state: int = 0

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Parameters:
            input: The bit sequence to be encoded.

        Returns:
            output: The encoded bit sequence.

        Examples:
            >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
            >>> encoder = komm.ConvolutionalStreamEncoder(convolutional_code)
            >>> encoder([1, 1, 1, 1])
            array([1, 1, 0, 1, 1, 0, 1, 0])
            >>> encoder([1, 1, 1, 1])
            array([1, 0, 1, 0, 1, 0, 1, 0])
        """
        input = np.asarray(input)
        n = self.convolutional_code.num_output_bits
        k = self.convolutional_code.num_input_bits
        fsm = self.convolutional_code.finite_state_machine()
        output_sequence, self.state = fsm.process(
            input_sequence=bits_to_int(input.reshape(-1, k)),
            initial_state=self.state,
        )
        output = int_to_bits(output_sequence, width=n).ravel()
        return output
