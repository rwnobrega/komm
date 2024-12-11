from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from attrs import field, frozen
from typing_extensions import override

from .._error_control_block.BlockCode import BlockCode
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.decorators import vectorized_method
from .ConvolutionalCode import ConvolutionalCode
from .terminations import (
    DirectTruncation,
    TailBiting,
    TerminationStrategy,
    ZeroTermination,
)

TerminationMode = Literal["direct-truncation", "zero-termination", "tail-biting"]


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

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='direct-truncation')
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 1, 1]])
        >>> code.minimum_distance()
        2

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='zero-termination')
        >>> (code.length, code.dimension, code.redundancy)
        (8, 3, 5)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 1]])
        >>> code.minimum_distance()
        3

        >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='tail-biting')
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 1, 0, 0, 1, 1]])
        >>> code.minimum_distance()
        3
    """

    convolutional_code: ConvolutionalCode
    num_blocks: int
    mode: TerminationMode = field(default="zero-termination")

    def __attrs_post_init__(self):
        if self.mode not in TerminationMode.__args__:
            raise ValueError(
                f"mode '{self.mode}' is unknown\n"
                f"supported termination modes: {set(TerminationMode.__args__)}"
            )

    @cached_property
    def _strategy(self) -> TerminationStrategy:
        return {
            "direct-truncation": DirectTruncation,
            "zero-termination": ZeroTermination,
            "tail-biting": TailBiting,
        }[self.mode](self.convolutional_code, self.num_blocks)

    @property
    @override
    def length(self) -> int:
        return self._strategy.codeword_length()

    @property
    @override
    def dimension(self) -> int:
        return self.num_blocks * self.convolutional_code.num_input_bits

    @property
    @override
    def redundancy(self) -> int:
        return self.length - self.dimension

    @cached_property
    @override
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        return self._strategy.generator_matrix(self)

    @vectorized_method
    def _enc_mapping(self, u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        k0 = self.convolutional_code.num_input_bits
        n0 = self.convolutional_code.num_output_bits
        fsm = self.convolutional_code.finite_state_machine()
        u = self._strategy.pre_process_input(u)
        input_sequence = bits_to_int(u.reshape(-1, k0))
        initial_state = self._strategy.initial_state(input_sequence)
        output_sequence, _ = fsm.process(input_sequence, initial_state)
        v = int_to_bits(output_sequence, width=n0).ravel()
        return v

    @property
    @override
    def default_decoder(self) -> str:
        return "viterbi-hard"

    @classmethod
    @override
    def supported_decoders(cls) -> list[str]:
        return cls.__base__.supported_decoders() + ["viterbi-hard", "viterbi-soft", "bcjr"]  # type: ignore

    @cached_property
    def cache_bit(self) -> npt.NDArray[np.integer]:
        n0 = self.convolutional_code.num_output_bits
        return np.array([int_to_bits(y, width=n0) for y in range(2**n0)])

    @cached_property
    def cache_polar(self) -> npt.NDArray[np.integer]:
        return (-1) ** self.cache_bit
