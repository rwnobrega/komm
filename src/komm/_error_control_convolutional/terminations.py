from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_power

from .._error_control_convolutional import TerminatedConvolutionalCode
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.matrices import pseudo_inverse
from .ConvolutionalCode import ConvolutionalCode


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


@dataclass
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


@dataclass
class ZeroTermination(TerminationStrategy):
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


@dataclass
class TailBiting(TerminationStrategy):
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

    @cached_property
    def _zs_multiplier(self) -> npt.NDArray[np.integer]:
        h = self.num_blocks
        nu = self.convolutional_code.overall_constraint_length
        A_mat, _, _, _ = self.convolutional_code.state_space_representation()
        return pseudo_inverse(matrix_power(A_mat, h) + np.eye(nu, dtype=int) % 2)
