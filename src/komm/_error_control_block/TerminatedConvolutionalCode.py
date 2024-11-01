from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from attrs import frozen
from numpy.linalg import matrix_power

from .._algebra._util import right_inverse
from .._error_control_convolutional.ConvolutionalCode import ConvolutionalCode
from .._util import binlist2int, int2binlist, pack, unpack
from .BlockCode import BlockCode


@frozen
class TerminatedConvolutionalCode(BlockCode):
    r"""
    Terminated convolutional code. It is a [linear block code](/ref/BlockCode) obtained by terminating a $(n_0, k_0)$ [convolutional code](/ref/ConvolutionalCode). A total of $h$ information blocks (each containing $k_0$ information bits) is encoded. The dimension of the resulting block code is thus $k = h k_0$; its length depends on the termination mode employed. There are three possible termination modes:

    - **Direct truncation**. The encoder always starts at state $0$, and its output ends immediately after the last information block. The encoder may not necessarily end in state $0$. The resulting block code will have length $n = h n_0$.

    - **Zero termination**. The encoder always starts and ends at state $0$. To achieve this, a sequence of $k \mu$ tail bits is appended to the information bits, where $\mu$ is the memory order of the convolutional code. The resulting block code will have length $n = (h + \mu) n_0$.

    - **Tail-biting**. The encoder always starts and ends at the same state. To achieve this, the initial state of the encoder is chosen as a function of the information bits. The resulting block code will have length $n = h n_0$.

    For more details, see <cite>LC04, Sec. 12.7</cite> and <cite>WBR01</cite>.

    Attributes:
        convolutional_code: The convolutional code to be terminated.

        num_blocks: The number $h$ of information blocks.

        mode: The termination mode. It must be one of `'direct-truncation'` | `'zero-termination'` | `'tail-biting'`. The default value is `'zero-termination'`.

    Examples:
        >>> convolutional_code = komm.ConvolutionalCode([[0b1, 0b11]])

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='zero-termination')
        >>> (code.length, code.dimension, code.minimum_distance)
        (np.int64(8), 3, np.int64(3))
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 1]])

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='direct-truncation')
        >>> (code.length, code.dimension, code.minimum_distance)
        (6, 3, np.int64(2))
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 1, 1]])

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='tail-biting')
        >>> (code.length, code.dimension, code.minimum_distance)
        (6, 3, np.int64(3))
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 1, 0, 0, 1, 1]])
    """

    convolutional_code: ConvolutionalCode
    num_blocks: int
    mode: Literal["direct-truncation", "zero-termination", "tail-biting"] = (
        "zero-termination"
    )

    def __attrs_post_init__(self):
        if self.mode == "tail-biting":
            try:
                self.zs_multiplier
            except:
                raise ValueError(
                    "This convolutional code does not support tail-biting for this number of blocks"
                )

    @property
    def length(self):
        total_num_blocks = self.num_blocks
        if self.mode == "zero-termination":
            total_num_blocks += self.convolutional_code.memory_order
        return total_num_blocks * self.convolutional_code.num_output_bits

    @property
    def dimension(self):
        return self.num_blocks * self.convolutional_code.num_input_bits

    @cached_property
    def generator_matrix(self):
        convolutional_code = self.convolutional_code
        k0, n0 = convolutional_code.num_input_bits, convolutional_code.num_output_bits
        k, n = self.dimension, self.length
        generator_matrix = np.zeros((k, n), dtype=int)
        top_rows = np.apply_along_axis(self.enc_mapping, 1, np.eye(k0, k, dtype=int))
        for t in range(self.num_blocks):
            generator_matrix[k0 * t : k0 * (t + 1), :] = np.roll(
                top_rows, shift=n0 * t, axis=1
            )
            if self.mode == "direct-truncation":
                generator_matrix[k0 * t : k0 * (t + 1), : n0 * t] = 0
        return generator_matrix

    def enc_mapping(self, u: npt.ArrayLike) -> np.ndarray:
        convolutional_code = self.convolutional_code
        k0, n0, nu, fsm = (
            convolutional_code.num_input_bits,
            convolutional_code.num_output_bits,
            convolutional_code.overall_constraint_length,
            convolutional_code.finite_state_machine,
        )
        if self.mode == "direct-truncation":
            input_sequence = pack(u, width=k0)
            initial_state = 0
        elif self.mode == "zero-termination":
            tail = np.dot(u, self.tail_projector) % 2
            input_sequence = pack(np.concatenate([u, tail]), width=k0)
            initial_state = 0
        else:  # self.mode == "tail-biting"
            # See [WBR01, Sec III.B].
            input_sequence = pack(u, width=k0)
            _, zs_response = fsm.process(input_sequence, initial_state=0)
            initial_state = binlist2int(
                np.dot(int2binlist(zs_response, width=nu), self.zs_multiplier) % 2
            )

        output_sequence, _ = fsm.process(input_sequence, initial_state)
        v = unpack(output_sequence, width=n0)
        return v

    @property
    def default_decoder(self):
        return "viterbi_hard"

    @classmethod
    def supported_decoders(cls):
        return cls.__base__.supported_decoders() + ["viterbi_hard", "viterbi_soft", "bcjr"]  # type: ignore

    @cached_property
    def tail_projector(self) -> np.ndarray:
        if self.mode != "zero-termination":
            raise ValueError(
                "This property is only defined for mode='zero-termination'"
            )
        h = self.num_blocks
        mu = self.convolutional_code.memory_order
        A_mat = self.convolutional_code.state_matrix
        B_mat = self.convolutional_code.control_matrix
        AnB_message = np.vstack(
            [
                np.dot(B_mat, matrix_power(A_mat, j)) % 2
                for j in range(mu + h - 1, mu - 1, -1)
            ]
        )
        AnB_tail = np.vstack(
            [np.dot(B_mat, matrix_power(A_mat, j)) % 2 for j in range(mu - 1, -1, -1)]
        )
        return np.dot(AnB_message, right_inverse(AnB_tail)) % 2

    @cached_property
    def zs_multiplier(self) -> np.ndarray:
        # See [WBR01, eq. (4)].
        if self.mode != "tail-biting":
            raise ValueError("This property is only defined for mode='tail-biting'")
        h = self.num_blocks
        nu = self.convolutional_code.overall_constraint_length
        A_mat = self.convolutional_code.state_matrix
        return right_inverse(matrix_power(A_mat, h) + np.eye(nu, dtype=int) % 2)

    @cached_property
    def cache_bit(self) -> np.ndarray:
        n0 = self.convolutional_code.num_output_bits
        return np.array([int2binlist(y, width=n0) for y in range(2**n0)])

    @cached_property
    def cache_polar(self) -> np.ndarray:
        return (-1) ** self.cache_bit
