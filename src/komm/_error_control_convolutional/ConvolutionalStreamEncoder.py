from .._util import pack, unpack


class ConvolutionalStreamEncoder:
    r"""
    Convolutional stream encoder. Encode a bit stream using a given [convolutional code](/ref/ConvolutionalCode). The internal state of the encoder is maintained across each call.

    To invoke the encoder, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, convolutional_code, initial_state=0):
        r"""
        Constructor for the class.

        Parameters:

            convolutional_code (ConvolutionalCode): The convolutional code.

            initial_state (Optional[int]): Initial state of the encoder. The default value is `0`.

        Examples:

            >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
            >>> convolutional_encoder = komm.ConvolutionalStreamEncoder(convolutional_code)
            >>> convolutional_encoder([1, 1, 1, 1])
            array([1, 1, 0, 1, 1, 0, 1, 0])
            >>> convolutional_encoder([1, 1, 1, 1])
            array([1, 0, 1, 0, 1, 0, 1, 0])
        """
        self._convolutional_code = convolutional_code
        self._state = int(initial_state)

    def __call__(self, inp):
        n, k = self._convolutional_code.num_output_bits, self._convolutional_code.num_input_bits

        output_sequence, self._state = self._convolutional_code.finite_state_machine.process(
            input_sequence=pack(inp, width=k), initial_state=self._state
        )

        outp = unpack(output_sequence, width=n)
        return outp
