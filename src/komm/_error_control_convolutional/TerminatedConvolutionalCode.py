from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_power

from .._error_control_block import base
from .._error_control_convolutional import TerminatedConvolutionalCode
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.decorators import blockwise, vectorize
from .._util.matrices import null_matrix, pseudo_inverse
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
    def strategy(self) -> "TerminationStrategy":
        return {
            "direct-truncation": DirectTruncation,
            "zero-termination": ZeroTermination,
            "tail-biting": TailBiting,
        }[self.mode](self.convolutional_code, self.num_blocks)

    @cached_property
    def length(self) -> int:
        return self.strategy.codeword_length()

    @cached_property
    def dimension(self) -> int:
        return self.num_blocks * self.convolutional_code.num_input_bits

    @cached_property
    def redundancy(self) -> int:
        return self.length - self.dimension

    @cached_property
    def rate(self) -> float:
        return super().rate

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        return self.strategy.generator_matrix(self)

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
            u = self.strategy.pre_process_input(u)
            input_sequence = bits_to_int(u.reshape(-1, k0))
            initial_state = self.strategy.initial_state(input_sequence)
            output_sequence, _ = fsm.process(input_sequence, initial_state)
            v = int_to_bits(output_sequence, width=n0).ravel()
            return v

        return encode(input)

    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().project_word(input)

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


@dataclass
class TerminationStrategy(ABC):
    convolutional_code: "ConvolutionalCode"
    num_blocks: int

    @abstractmethod
    def initial_state(self, input_sequence: npt.ArrayLike) -> int: ...

    @abstractmethod
    def pre_process_input(
        self, input_bits: npt.ArrayLike
    ) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def codeword_length(self) -> int: ...

    @abstractmethod
    def generator_matrix(
        self, code: TerminatedConvolutionalCode
    ) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def viterbi_post_process_output(
        self, xs_hat: npt.NDArray[np.integer], final_metrics: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def bcjr_initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    @abstractmethod
    def bcjr_post_process_output(
        self, posteriors: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]: ...


def _base_generator_matrix(
    code: TerminatedConvolutionalCode,
    convolutional_code: ConvolutionalCode,
    num_blocks: int,
) -> npt.NDArray[np.integer]:
    k0 = convolutional_code.num_input_bits
    n0 = convolutional_code.num_output_bits
    k, n = code.dimension, code.length
    generator_matrix = np.zeros((k, n), dtype=int)
    top_rows = np.apply_along_axis(code.encode, 1, np.eye(k0, k, dtype=int))
    for t in range(num_blocks):
        generator_matrix[k0 * t : k0 * (t + 1), :] = np.roll(top_rows, n0 * t, 1)
    return generator_matrix


class DirectTruncation(TerminationStrategy):
    def initial_state(self, input_sequence: npt.ArrayLike) -> int:
        return 0

    def pre_process_input(self, input_bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return np.asarray(input_bits)

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        return h * n0

    def generator_matrix(
        self, code: TerminatedConvolutionalCode
    ) -> npt.NDArray[np.integer]:
        h = self.num_blocks
        k0 = self.convolutional_code.num_input_bits
        n0 = self.convolutional_code.num_output_bits
        generator_matrix = _base_generator_matrix(code, self.convolutional_code, h)
        for t in range(1, h):
            generator_matrix[k0 * t : k0 * (t + 1), : n0 * t] = 0
        return generator_matrix

    def viterbi_post_process_output(
        self, xs_hat: npt.NDArray[np.integer], final_metrics: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        s_hat = np.argmin(final_metrics)
        x_hat = xs_hat[:, s_hat]
        return x_hat

    def bcjr_initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        initial_distribution = np.eye(1, num_states, 0)
        final_distribution = np.ones(num_states) / num_states
        return initial_distribution, final_distribution

    def bcjr_post_process_output(
        self, posteriors: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return posteriors


class ZeroTermination(TerminationStrategy):
    @cached_property
    def _tail_projector(self) -> npt.NDArray[np.integer]:
        h = self.num_blocks
        mu = self.convolutional_code.memory_order
        A_mat, B_mat, _, _ = self.convolutional_code.state_space_representation()
        AnB_message = np.vstack(
            [B_mat @ matrix_power(A_mat, j) % 2 for j in range(mu + h - 1, mu - 1, -1)]
        )
        AnB_tail = np.vstack(
            [B_mat @ matrix_power(A_mat, j) % 2 for j in range(mu - 1, -1, -1)]
        )
        return AnB_message @ pseudo_inverse(AnB_tail) % 2

    def initial_state(self, input_sequence: npt.ArrayLike) -> int:
        return 0

    def pre_process_input(self, input_bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        input_bits = np.asarray(input_bits)
        tail = input_bits @ self._tail_projector % 2
        return np.concatenate([input_bits, tail])

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        m = self.convolutional_code.memory_order
        return (h + m) * n0

    def generator_matrix(
        self, code: TerminatedConvolutionalCode
    ) -> npt.NDArray[np.integer]:
        return _base_generator_matrix(code, self.convolutional_code, self.num_blocks)

    def viterbi_post_process_output(
        self, xs_hat: npt.NDArray[np.integer], final_metrics: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        mu = self.convolutional_code.memory_order
        x_hat = xs_hat[:, 0][:-mu]
        return x_hat

    def bcjr_initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        initial_distribution = np.eye(1, num_states, 0)
        final_distribution = np.eye(1, num_states, 0)
        return initial_distribution, final_distribution

    def bcjr_post_process_output(
        self, posteriors: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mu = self.convolutional_code.memory_order
        return posteriors[:-mu]


class TailBiting(TerminationStrategy):
    @cached_property
    def _zs_multiplier(self) -> npt.NDArray[np.integer]:
        h = self.num_blocks
        nu = self.convolutional_code.overall_constraint_length
        A_mat, _, _, _ = self.convolutional_code.state_space_representation()
        return pseudo_inverse((matrix_power(A_mat, h) + np.eye(nu, dtype=int)) % 2)

    def initial_state(self, input_sequence: npt.ArrayLike) -> int:
        fsm = self.convolutional_code.finite_state_machine()
        nu = self.convolutional_code.overall_constraint_length
        _, zs_response = fsm.process(input_sequence, initial_state=0)
        zs_response = int_to_bits(zs_response, width=nu)
        initial_state = bits_to_int(zs_response @ self._zs_multiplier % 2)
        assert isinstance(initial_state, int)
        return initial_state

    def pre_process_input(self, input_bits: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return np.asarray(input_bits)

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        return h * n0

    def generator_matrix(
        self, code: TerminatedConvolutionalCode
    ) -> npt.NDArray[np.integer]:
        return _base_generator_matrix(code, self.convolutional_code, self.num_blocks)

    def viterbi_post_process_output(
        self, xs_hat: npt.NDArray[np.integer], final_metrics: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        raise NotImplementedError

    def bcjr_initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        raise NotImplementedError

    def bcjr_post_process_output(
        self, posteriors: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        raise NotImplementedError
