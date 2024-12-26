from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._error_control_block import base
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.decorators import blockwise, vectorize
from .._util.matrices import null_matrix, pseudo_inverse
from . import terminations
from .ConvolutionalCode import ConvolutionalCode

TerminationMode = Literal["direct-truncation", "zero-termination", "tail-biting"]


@dataclass(eq=False)
class TerminatedConvolutionalCode(base.BlockCode):
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
    mode: TerminationMode = "zero-termination"

    def __post_init__(self):
        if not self.mode in TerminationMode.__args__:
            raise ValueError(
                f"mode '{self.mode}' is unknown\n"
                f"supported termination modes: {set(TerminationMode.__args__)}"
            )

    @cached_property
    def _strategy(self) -> terminations.TerminationStrategy:
        return {
            "direct-truncation": terminations.DirectTruncation,
            "zero-termination": terminations.ZeroTermination,
            "tail-biting": terminations.TailBiting,
        }[self.mode](self.convolutional_code, self.num_blocks)

    @property
    def length(self) -> int:
        return self._strategy.codeword_length()

    @property
    def dimension(self) -> int:
        return self.num_blocks * self.convolutional_code.num_input_bits

    @property
    def redundancy(self) -> int:
        return self.length - self.dimension

    @property
    def rate(self) -> float:
        return super().rate

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        return self._strategy.generator_matrix(self)

    @cached_property
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        return pseudo_inverse(self.generator_matrix)

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        return null_matrix(self.generator_matrix)

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        k0 = self.convolutional_code.num_input_bits
        n0 = self.convolutional_code.num_output_bits
        fsm = self.convolutional_code.finite_state_machine()

        @blockwise(self.dimension)
        @vectorize
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            u = self._strategy.pre_process_input(u)
            input_sequence = bits_to_int(u.reshape(-1, k0))
            initial_state = self._strategy.initial_state(input_sequence)
            output_sequence, _ = fsm.process(input_sequence, initial_state)
            v = int_to_bits(output_sequence, width=n0).ravel()
            return v

        return encode(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().inverse_encode(input)

    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().check(input)

    @cache
    def codewords(self) -> npt.NDArray[np.integer]:
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        return super().coset_leaders()

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        return super().covering_radius()
