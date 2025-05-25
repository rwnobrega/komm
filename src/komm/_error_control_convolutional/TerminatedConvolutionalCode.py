from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_power

from .._error_control_block import base
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
        >>> code = komm.TerminatedConvolutionalCode(
        ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
        ...     num_blocks=3,
        ...     mode='direct-truncation',
        ... )
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 1, 1]])
        >>> code.minimum_distance()
        2

        >>> code = komm.TerminatedConvolutionalCode(
        ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
        ...     num_blocks=3,
        ...     mode='zero-termination',
        ... )
        >>> (code.length, code.dimension, code.redundancy)
        (8, 3, 5)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 1]])
        >>> code.minimum_distance()
        3

        >>> code = komm.TerminatedConvolutionalCode(
        ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
        ...     num_blocks=3,
        ...     mode='tail-biting',
        ... )
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
        return self.encode(np.eye(self.dimension))

    @cached_property
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        return pseudo_inverse(self.generator_matrix)

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        return null_matrix(self.generator_matrix)

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.dimension)
        @vectorize
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            v, _ = self.convolutional_code.encode_with_state(
                input=self.strategy.pre_process_input(u),
                initial_state=self.strategy.initial_state(u),
            )
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
    def pre_process_input(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def codeword_length(self) -> int: ...

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


class DirectTruncation(TerminationStrategy):
    def pre_process_input(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return np.asarray(input)

    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        σ = self.convolutional_code.degree
        return np.zeros(σ, dtype=int)

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        return h * n0

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
        # See [WBR01, eq. (3)]. Set x_0 = x_t = 0, and t = h + μ.
        h = self.num_blocks
        μ = self.convolutional_code.memory_order
        σ = self.convolutional_code.degree
        A_mat, B_mat, _, _ = self.convolutional_code.state_space_representation()
        A_pow = [np.eye(σ, dtype=int)]
        for _ in range(1, h + μ):
            A_pow.append((A_pow[-1] @ A_mat) % 2)
        M_info = np.vstack([B_mat @ A_pow[j] % 2 for j in range(h + μ - 1, μ - 1, -1)])
        M_tail = np.vstack([B_mat @ A_pow[j] % 2 for j in range(μ - 1, -1, -1)])
        return M_info @ pseudo_inverse(M_tail) % 2

    def pre_process_input(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        input = np.asarray(input)
        tail = input @ self._tail_projector % 2
        return np.concatenate([input, tail])

    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        σ = self.convolutional_code.degree
        return np.zeros(σ, dtype=int)

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        μ = self.convolutional_code.memory_order
        return (h + μ) * n0

    def viterbi_post_process_output(
        self, xs_hat: npt.NDArray[np.integer], final_metrics: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        μ = self.convolutional_code.memory_order
        x_hat = xs_hat[:, 0][:-μ]
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
        μ = self.convolutional_code.memory_order
        return posteriors[:-μ]


class TailBiting(TerminationStrategy):
    @cached_property
    def _zs_multiplier(self) -> npt.NDArray[np.integer]:
        # See [WBR01, eq. (4)].
        h = self.num_blocks
        σ = self.convolutional_code.degree
        A_mat, _, _, _ = self.convolutional_code.state_space_representation()
        return pseudo_inverse((matrix_power(A_mat, h) + np.eye(σ, dtype=int)) % 2)

    def pre_process_input(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return np.asarray(input)

    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        σ = self.convolutional_code.degree
        zero_state = np.zeros(σ, dtype=int)
        _, state = self.convolutional_code.encode_with_state(input, zero_state)
        return state @ self._zs_multiplier % 2

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        return h * n0

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
