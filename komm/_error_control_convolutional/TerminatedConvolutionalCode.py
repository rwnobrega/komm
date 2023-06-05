import functools

import numpy as np

from .._algebra.util import right_inverse
from .._aux import tag
from .._error_control_block.BlockCode import BlockCode
from .._util import binlist2int, int2binlist, pack, unpack


class TerminatedConvolutionalCode(BlockCode):
    r"""
    Terminated convolutional code. It is a linear block code (:class:`BlockCode`) obtained by terminating a :math:`(n_0, k_0)` convolutional code (:class:`ConvolutionalCode`). A total of :math:`h` information blocks (each containing :math:`k_0` information bits) is encoded. The dimension of the resulting block code is thus :math:`k = h k_0`; its length depends on the termination mode employed. There are three possible termination modes:

    - **Direct truncation**. The encoder always starts at state :math:`0`, and its output ends immediately after the last information block. The encoder may not necessarily end in state :math:`0`. The resulting block code will have length :math:`n = h n_0`.

    - **Zero termination**. The encoder always starts and ends at state :math:`0`. To achieve this, a sequence of :math:`k \mu` tail bits is appended to the information bits, where :math:`\mu` is the memory order of the convolutional code. The resulting block code will have length :math:`n = (h + \mu) n_0`.

    - **Tail-biting**. The encoder always starts and ends at the same state. To achieve this, the initial state of the encoder is chosen as a function of the information bits. The resulting block code will have length :math:`n = h n_0`.

    .. rubric:: Decoding methods

    [[decoding_methods]]

    References:

        1. :cite:`Lin.Costello.04`
        2. :cite:`Weiss.01`
    """

    def __init__(self, convolutional_code, num_blocks, mode="zero-termination"):
        r"""
        Constructor for the class.

        Parameters:

            convolutional_code (:obj:`ConvolutionalCode`): The convolutional code to be terminated.

            num_blocks (:obj:`int`): The number :math:`h` of information blocks.

            mode (:obj:`str`, optional): The termination mode. It must be one of :code:`'direct-truncation'` | :code:`'zero-termination'` | :code:`'tail-biting'`. The default value is :code:`'zero-termination'`.

        Examples:

            >>> convolutional_code = komm.ConvolutionalCode([[0b1, 0b11]])
            >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='zero-termination')
            >>> (code.length, code.dimension, code.minimum_distance)
            (8, 3, 3)
            >>> code.generator_matrix
            array([[1, 1, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 1]])

            >>> convolutional_code = komm.ConvolutionalCode([[0b1, 0b11]])
            >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='direct-truncation')
            >>> (code.length, code.dimension, code.minimum_distance)
            (6, 3, 2)
            >>> code.generator_matrix
            array([[1, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1],
                   [0, 0, 0, 0, 1, 1]])

            >>> convolutional_code = komm.ConvolutionalCode([[0b1, 0b11]])
            >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='tail-biting')
            >>> (code.length, code.dimension, code.minimum_distance)
            (6, 3, 3)
            >>> code.generator_matrix
            array([[1, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1],
                   [0, 1, 0, 0, 1, 1]])
        """
        self._convolutional_code = convolutional_code
        self._mode = mode
        self._num_blocks = h = num_blocks

        if mode not in ["direct-truncation", "zero-termination", "tail-biting"]:
            raise ValueError("Parameter 'mode' must be in {'direct-truncation', 'zero-termination', 'tail-biting'}")

        k0, n0 = convolutional_code.num_input_bits, convolutional_code.num_output_bits
        nu, mu = convolutional_code.overall_constraint_length, convolutional_code.memory_order

        self._dimension = h * k0
        if mode in ["direct-truncation", "tail-biting"]:
            self._length = h * n0
        elif mode == "zero-termination":
            self._length = (h + mu) * n0
        self._redundancy = self._length - self._dimension

        A, B = convolutional_code.state_matrix, convolutional_code.control_matrix

        if mode == "zero-termination":
            AnB_message = np.concatenate(
                [np.dot(B, np.linalg.matrix_power(A, j)) % 2 for j in range(mu + h - 1, mu - 1, -1)], axis=0
            )
            AnB_tail = np.concatenate(
                [np.dot(B, np.linalg.matrix_power(A, j)) % 2 for j in range(mu - 1, -1, -1)], axis=0
            )
            self._tail_projector = np.dot(AnB_message, right_inverse(AnB_tail)) % 2
        elif mode == "tail-biting":
            try:
                M = (np.linalg.matrix_power(A, h) + np.eye(nu, dtype=int)) % 2
                self._M_inv = right_inverse(M)
            except:
                raise ValueError("This convolutional code does not support tail-biting for this number of blocks")

        cache_bit = np.array([int2binlist(y, width=n0) for y in range(2**n0)])
        self._metric_function_viterbi_hard = lambda y, z: np.count_nonzero(cache_bit[y] != z)
        self._metric_function_viterbi_soft = lambda y, z: np.dot(cache_bit[y], z)
        cache_polar = (-1) ** cache_bit
        self._metric_function_bcjr = lambda SNR, y, z: 2.0 * SNR * np.dot(cache_polar[y], z)

    def __repr__(self):
        args = "convolutional_code={}, num_blocks={}, mode='{}'".format(
            self._convolutional_code, self._num_blocks, self._mode
        )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def num_blocks(self):
        r"""
        The number :math:`h` of information blocks of the terminated convolutional code. This property is read-only.
        """
        return self._num_blocks

    @property
    def mode(self):
        r"""
        The termination mode of the terminated convolutional code. This property is read-only.
        """
        return self._mode

    @functools.cached_property
    def generator_matrix(self):
        k0, n0 = self._convolutional_code.num_input_bits, self._convolutional_code.num_output_bits
        generator_matrix = np.zeros((self._dimension, self._length), dtype=int)
        top_rows = np.apply_along_axis(self._encode_finite_state_machine, 1, np.eye(k0, self._dimension, dtype=int))
        for t in range(self._num_blocks):
            generator_matrix[k0 * t : k0 * (t + 1), :] = np.roll(top_rows, shift=n0 * t, axis=1)
            if self._mode == "direct-truncation":
                generator_matrix[k0 * t : k0 * (t + 1), : n0 * t] = 0
        return generator_matrix

    def _encode_finite_state_machine(self, message):
        convolutional_code = self._convolutional_code
        k0, n0, nu = (
            convolutional_code.num_input_bits,
            convolutional_code.num_output_bits,
            convolutional_code.overall_constraint_length,
        )

        if self._mode == "direct-truncation":
            input_sequence = pack(message, width=k0)
            initial_state = 0
        elif self._mode == "zero-termination":
            tail = np.dot(message, self._tail_projector) % 2
            input_sequence = pack(np.concatenate([message, tail]), width=k0)
            initial_state = 0
        else:  # self._mode == "tail-biting"
            # See Weiss.01.
            input_sequence = pack(message, width=k0)
            _, zero_state_solution = convolutional_code.finite_state_machine.process(input_sequence, initial_state=0)
            initial_state = binlist2int(np.dot(int2binlist(zero_state_solution, width=nu), self._M_inv) % 2)

        output_sequence, _ = convolutional_code.finite_state_machine.process(input_sequence, initial_state)
        codeword = unpack(output_sequence, width=n0)
        return codeword

    def _default_encoder(self):
        return "finite_state_machine"

    def _helper_decode_viterbi(self, recvword, metric_function):
        convolutional_code = self._convolutional_code
        k0, n0, mu = (
            convolutional_code.num_input_bits,
            convolutional_code.num_output_bits,
            convolutional_code.memory_order,
        )
        num_states = convolutional_code.finite_state_machine.num_states

        if self._mode in ["direct-truncation", "zero-termination"]:
            initial_metrics = np.full(num_states, fill_value=np.inf)
            initial_metrics[0] = 0.0
        else:  # self._mode == "tail-biting"
            raise NotImplementedError("Viterbi algorithm not implemented for 'tail-biting'")

        input_sequences_hat, final_metrics = convolutional_code.finite_state_machine.viterbi(
            observed_sequence=np.reshape(recvword, newshape=(-1, n0)),
            metric_function=metric_function,
            initial_metrics=initial_metrics,
        )

        if self._mode == "direct-truncation":
            final_state_hat = np.argmin(final_metrics)
            input_sequence_hat = input_sequences_hat[:, final_state_hat]
        elif self._mode == "zero-termination":
            input_sequence_hat = input_sequences_hat[:, 0][:-mu]
        else:  # self._mode == "tail-biting"
            raise NotImplementedError("Viterbi algorithm not implemented for 'tail-biting'")

        message_hat = unpack(input_sequence_hat, width=k0)
        return message_hat

    @tag(name="Viterbi (hard-decision)", input_type="hard", target="message")
    def _decode_viterbi_hard(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_hard)

    @tag(name="Viterbi (soft-decision)", input_type="soft", target="message")
    def _decode_viterbi_soft(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_soft)

    @tag(name="BCJR", input_type="soft", target="message")
    def _decode_bcjr(self, recvword, output_type="hard", SNR=1.0):
        convolutional_code = self._convolutional_code
        k0, n0, mu = (
            convolutional_code.num_input_bits,
            convolutional_code.num_output_bits,
            convolutional_code.memory_order,
        )
        num_states = convolutional_code.finite_state_machine.num_states

        if self._mode == "direct-truncation":
            initial_state_distribution = np.eye(1, num_states, 0)
            final_state_distribution = np.ones(num_states) / num_states
        elif self._mode == "zero-termination":
            initial_state_distribution = np.eye(1, num_states, 0)
            final_state_distribution = np.eye(1, num_states, 0)
        else:
            raise NotImplementedError("BCJR algorithm not implemented for 'tail-biting'")

        input_posteriors = convolutional_code.finite_state_machine.forward_backward(
            observed_sequence=np.reshape(recvword, newshape=(-1, n0)),
            metric_function=lambda y, z: self._metric_function_bcjr(SNR, y, z),
            initial_state_distribution=initial_state_distribution,
            final_state_distribution=final_state_distribution,
        )

        if self._mode == "zero-termination":
            input_posteriors = input_posteriors[:-mu]

        if output_type == "soft":
            return np.log(input_posteriors[:, 0] / input_posteriors[:, 1])
        elif output_type == "hard":
            input_sequence_hat = np.argmax(input_posteriors, axis=1)
            return unpack(input_sequence_hat, width=k0)

    def _default_decoder(self, dtype):
        if dtype == int:
            return "viterbi_hard"
        elif dtype == float:
            return "viterbi_soft"
