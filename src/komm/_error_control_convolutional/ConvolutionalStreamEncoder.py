import numpy as np
import numpy.typing as npt
from attrs import field, mutable

from .._util.bit_operations import bits_to_int, int_to_bits
from .ConvolutionalCode import ConvolutionalCode


@mutable
class ConvolutionalStreamEncoder:
    r"""
    Convolutional stream encoder. Encode a bit stream using a given [convolutional code](/ref/ConvolutionalCode). The internal state of the encoder is maintained across each call.

    Attributes:
        convolutional_code: The convolutional code.
        state: The current state of the encoder. The default value is `0`.

    Parameters: Input:
        in0 (Array1D[int]): The bit sequence to be encoded.

    Parameters: Output:
        out0 (Array1D[int]): The encoded bit sequence.

    Examples:
        >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
        >>> convolutional_encoder = komm.ConvolutionalStreamEncoder(convolutional_code)
        >>> convolutional_encoder([1, 1, 1, 1])
        array([1, 1, 0, 1, 1, 0, 1, 0])
        >>> convolutional_encoder([1, 1, 1, 1])
        array([1, 0, 1, 0, 1, 0, 1, 0])
    """

    convolutional_code: ConvolutionalCode
    state: int = field(default=0)

    def __call__(self, in0: npt.ArrayLike) -> npt.NDArray[np.int_]:
        in0 = np.asarray(in0)
        n = self.convolutional_code.num_output_bits
        k = self.convolutional_code.num_input_bits
        fsm = self.convolutional_code.finite_state_machine()
        output_sequence, self.state = fsm.process(
            input_sequence=bits_to_int(in0.reshape(-1, k)),
            initial_state=self.state,
        )
        out0 = int_to_bits(output_sequence, width=n).ravel()
        return out0
