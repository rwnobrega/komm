import numpy as np

from .._util import int2binlist, unpack


class ConvolutionalStreamDecoder:
    r"""
    Convolutional stream decoder using Viterbi algorithm. Decode a (hard or soft) bit stream given a [convolutional code](/ref/ConvolutionalCode), assuming a traceback length (path memory) of $\tau$. At time $t$, the decoder chooses the path survivor with best metric at time $t - \tau$ and outputs the corresponding information bits. The output stream has a delay equal to $k \tau$, where $k$ is the number of input bits of the convolutional code. As a rule of thumb, the traceback length is chosen as $\tau = 5\mu$, where $\mu$ is the memory order of the convolutional code.

    To invoke the decoder, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, convolutional_code, traceback_length, initial_state=0, input_type="hard"):
        r"""
        Constructor for the class.

        Parameters:

            convolutional_code (ConvolutionalCode): The convolutional code.

            traceback_length (int): The traceback length (path memory) $\tau$ of the decoder.

            initial_state (Optional[int]): Initial state of the encoder. The default value is `0`.

        Examples:

            >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
            >>> convolutional_decoder = komm.ConvolutionalStreamDecoder(convolutional_code, traceback_length=10)
            >>> convolutional_decoder([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            >>> convolutional_decoder(np.zeros(2*10, dtype=int))
            array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        """
        self._convolutional_code = convolutional_code
        self._traceback_length = int(traceback_length)
        self._initial_state = int(initial_state)
        self._input_type = input_type

        n = convolutional_code.num_output_bits
        num_states = convolutional_code.finite_state_machine.num_states

        self._memory = {}
        self._memory["metrics"] = np.full((num_states, traceback_length + 1), fill_value=np.inf)
        self._memory["metrics"][initial_state, -1] = 0.0
        self._memory["paths"] = np.zeros((num_states, traceback_length + 1), dtype=int)

        cache_bit = np.array([int2binlist(y, width=n) for y in range(2**n)])
        self._metric_function_hard = lambda y, z: np.count_nonzero(cache_bit[y] != z)
        self._metric_function_soft = lambda y, z: np.dot(cache_bit[y], z)

    def __call__(self, inp):
        n, k = self._convolutional_code.num_output_bits, self._convolutional_code.num_input_bits

        input_sequence_hat = self._convolutional_code.finite_state_machine.viterbi_streaming(
            observed_sequence=np.reshape(inp, newshape=(-1, n)),
            metric_function=getattr(self, "_metric_function_" + self._input_type),
            memory=self._memory,
        )

        outp = unpack(input_sequence_hat, width=k)
        return outp
