from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_power

from .. import abc
from .._util.decorators import blockwise
from .._util.matrices import matmul, null_matrix, pseudo_inverse, rank
from ..types import Array1D, Array2D

TerminationMode = Literal["direct-truncation", "zero-termination", "tail-biting"]


@dataclass(eq=False)
class TerminatedConvolutionalCode(abc.BlockCode):
    r"""
    Terminated convolutional code. It is a [linear block code](/ref/BlockCode) obtained by terminating a $(n_0, k_0)$ [convolutional code](/ref/ConvolutionalCode). A total of $h$ information blocks (each containing $k_0$ information bits) is encoded. The dimension of the resulting block code is thus $k = h k_0$; its length depends on the termination mode employed. There are three possible termination modes:

    - **Direct truncation**. The encoder always starts at state $0$, and its output ends immediately after the last information block. The encoder may not necessarily end in state $0$. The resulting block code will have length $n = h n_0$.

    - **Zero termination**. The encoder always starts and ends at state $0$. To achieve this, a sequence of $k \mu$ tail bits is appended to the information bits, where $\mu$ is the memory order of the convolutional code. The resulting block code will have length $n = (h + \mu) n_0$.

    - **Tail-biting**. The encoder always starts and ends at the same state. To achieve this, the initial state of the encoder is chosen as a function of the information bits. This is possible only if $A^h + I$ is invertible, where $A$ is the state matrix of the convolutional code; this always holds for codes without feedback. The resulting block code will have length $n = h n_0$.

    The code may also be *punctured*: after termination, the codeword bits are kept or deleted according to a *puncturing matrix* $\mathbf{P} \in \mathbb{B}^{n_0 \times T}$, in which the element at row $i$ and column $t$ tells whether the $i$-th output bit of block $t \bmod T$ is kept (`1`) or deleted (`0`), tail blocks included. The lengths given above are then denoted by $n_\mathrm{u}$, the length of the *unpunctured code*, and the length $n$ of the punctured code is the number of kept positions. The puncturing must not delete any nonzero codeword, so that the dimension $k$ is preserved.

    For more details, see <cite>LC04, Sec. 12.7</cite> and <cite>WBR01</cite>.

    Parameters:
        convolutional_code: The convolutional code to be terminated.

        num_blocks: The number $h$ of information blocks.

        mode: The termination mode. It must be one of `'direct-truncation'` | `'zero-termination'` | `'tail-biting'`. The default value is `'zero-termination'`.

        puncturing_matrix: The puncturing matrix $\mathbf{P}$. Must be an $n_0 \times T$ array of bits with at least one `1`, where the period $T$ divides the number $n_\mathrm{u} / n_0$ of trellis sections. The default value corresponds to no puncturing.

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

        >>> code = komm.TerminatedConvolutionalCode(
        ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
        ...     num_blocks=3,
        ...     mode='zero-termination',
        ...     puncturing_matrix=[[1, 1], [1, 0]],
        ... )
        >>> (code.length, code.dimension, code.redundancy)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 0, 1, 1, 0]])
        >>> code.minimum_distance()
        2
    """

    convolutional_code: abc.ConvolutionalCode
    num_blocks: int
    mode: TerminationMode = "zero-termination"
    puncturing_matrix: npt.ArrayLike | None = None

    def __post_init__(self):
        if not self.mode in TerminationMode.__args__:
            raise ValueError(
                f"mode '{self.mode}' is unknown\n"
                f"supported termination modes: {set(TerminationMode.__args__)}"
            )
        self.strategy = {
            "direct-truncation": DirectTruncation,
            "zero-termination": ZeroTermination,
            "tail-biting": TailBiting,
        }[self.mode](self.convolutional_code, self.num_blocks)
        n0 = self.convolutional_code.num_output_bits
        num_sections = self.strategy.codeword_length() // n0
        P_mat = np.ones((n0, 1), dtype=int)
        if self.puncturing_matrix is not None:
            P_mat = np.asarray(self.puncturing_matrix)
        if not P_mat.ndim == 2 or not np.isin(P_mat, [0, 1]).all():
            raise ValueError("'puncturing_matrix' must be a 2D-array of bits")
        if not P_mat.shape[0] == n0:
            raise ValueError("'puncturing_matrix' must have one row per output bit")
        if not np.any(P_mat):
            raise ValueError("'puncturing_matrix' must keep at least one bit")
        period = P_mat.shape[1]
        if not num_sections % period == 0:
            raise ValueError(
                "'puncturing_matrix' period must divide number of sections"
            )
        self._kept = np.flatnonzero(np.tile(P_mat.T.ravel(), num_sections // period))

    @cached_property
    def length(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.length
            6
        """
        return self._kept.size

    @cached_property
    def dimension(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.dimension
            3
        """
        return self.num_blocks * self.convolutional_code.num_input_bits

    @cached_property
    def redundancy(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.redundancy
            3
        """
        return self.length - self.dimension

    @cached_property
    def rate(self) -> float:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.rate
            0.5
        """
        return super().rate

    @cached_property
    def generator_matrix(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.generator_matrix
            array([[1, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1],
                   [0, 1, 0, 0, 1, 1]])
        """
        return self.encode(np.eye(self.dimension))

    @cached_property
    def check_matrix(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.check_matrix
            array([[1, 0, 1, 1, 0, 0],
                   [1, 1, 0, 0, 1, 0],
                   [1, 1, 1, 0, 0, 1]])
        """
        return null_matrix(self.generator_matrix)

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            <span style="margin-left: 1.5em; font-style: italic;">
            See [`BlockCode.encode`](/ref/BlockCode#encode) for examples.
            </span>
        """

        @blockwise(self.dimension)
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            v, _ = self.convolutional_code.encode_with_state(
                input=self.strategy.pre_process_input(u),
                initial_state=self.strategy.initial_state(u),
            )
            return v[..., self._kept]

        return encode(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            <span style="margin-left: 1.5em; font-style: italic;">
            See [`BlockCode.inverse_encode`](/ref/BlockCode#inverse_encode) for examples.
            </span>
        """
        return super().inverse_encode(input)

    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            <span style="margin-left: 1.5em; font-style: italic;">
            See [`BlockCode.check`](/ref/BlockCode#check) for examples.
            </span>
        """
        return super().check(input)

    def depuncture(
        self,
        input: npt.ArrayLike,
        fill_value: float,
    ) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Undoes the puncturing, by inserting a given value at the deleted positions. This method takes one or more sequences of received words and returns their corresponding sequences in the coordinates of the unpunctured code.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

            fill_value: The value to insert at the deleted positions, such as `0.0` for L-values or `2` for erasures.

        Returns:
            output: The output sequence(s). Has the same shape as the input, but with the last dimension expanded by a factor of $n_\mathrm{u} / n$.

        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='zero-termination',
            ...     puncturing_matrix=[[1, 1], [1, 0]],
            ... )
            >>> code.depuncture([1, 1, 1, 0, 0, 0], 2)
            array([1, 1, 1, 2, 0, 0, 0, 2])
            >>> code.depuncture([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1], 0.0)
            array([-0.8, -0.1, -1. ,  0. ,  0.5,  1.8, -1.1,  0. ])
        """
        n_u = self.strategy.codeword_length()

        @blockwise(self.length)
        def depuncture(r: npt.NDArray[np.integer | np.floating]):
            dtype = np.result_type(r, fill_value)
            v = np.full((*r.shape[:-1], n_u), fill_value, dtype=dtype)
            v[..., self._kept] = r
            return v

        return depuncture(input)

    @cache
    def codewords(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.codewords()
            array([[0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 1],
                   [0, 1, 0, 0, 1, 1],
                   [1, 0, 0, 1, 1, 1],
                   [0, 1, 1, 1, 1, 0],
                   [1, 0, 1, 0, 1, 0]])
        """
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.codeword_weight_distribution()
            array([1, 0, 0, 4, 3, 0, 0])
        """
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.minimum_distance()
            3
        """
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> Array2D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.coset_leaders()
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0]])
        """
        return super().coset_leaders()

    @cache
    def coset_leader_weight_distribution(self) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.coset_leader_weight_distribution()
            array([1, 6, 1, 0, 0, 0, 0])
        """
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.packing_radius()
            1
        """
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b1, 0b11]]),
            ...     num_blocks=3,
            ...     mode='tail-biting',
            ... )
            >>> code.covering_radius()
            2
        """
        return super().covering_radius()


@dataclass
class TerminationStrategy(ABC):
    convolutional_code: abc.ConvolutionalCode
    num_blocks: int

    @abstractmethod
    def pre_process_input(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def codeword_length(self) -> int: ...

    @abstractmethod
    def initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


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

    def initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        initial_distribution = np.eye(1, num_states, 0)
        final_distribution = np.ones(num_states) / num_states
        return initial_distribution, final_distribution


class ZeroTermination(TerminationStrategy):
    @cached_property
    def _tail_projector(self) -> npt.NDArray[np.integer]:
        # See [WBR01, eq. (3)]. Set x_0 = x_t = 0, and t = h + μ.
        h = self.num_blocks
        μ = self.convolutional_code.memory_order
        if μ == 0:
            k0 = self.convolutional_code.num_input_bits
            return np.zeros((h * k0, 0), dtype=int)
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
        tail = matmul(input, self._tail_projector)
        return np.concatenate([input, tail], axis=-1)

    def initial_state(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        σ = self.convolutional_code.degree
        return np.zeros(σ, dtype=int)

    def codeword_length(self) -> int:
        h = self.num_blocks
        n0 = self.convolutional_code.num_output_bits
        μ = self.convolutional_code.memory_order
        return (h + μ) * n0

    def initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        initial_distribution = np.eye(1, num_states, 0)
        final_distribution = np.eye(1, num_states, 0)
        return initial_distribution, final_distribution


@dataclass
class TailBiting(TerminationStrategy):
    def __post_init__(self) -> None:
        # See [WBR01, eq. (4)].
        h = self.num_blocks
        σ = self.convolutional_code.degree
        A_mat, _, _, _ = self.convolutional_code.state_space_representation()
        matrix = (matrix_power(A_mat, h) + np.eye(σ, dtype=int)) % 2
        if rank(matrix) < σ:
            raise ValueError("tail-biting is impossible for this code and 'num_blocks'")
        self._zs_multiplier = pseudo_inverse(matrix)

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

    def initial_final_distributions(
        self, num_states: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        raise NotImplementedError
