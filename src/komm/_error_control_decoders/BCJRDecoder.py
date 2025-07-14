from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_convolutional.TerminatedConvolutionalCode import (
    TerminatedConvolutionalCode,
)
from .._labelings.NaturalLabeling import NaturalLabeling
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, vectorize, with_pbar
from .util import get_pbar


@dataclass
class BCJRDecoder(abc.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Bahl–Cocke–Jelinek–Raviv (BCJR) decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.6</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        output_type: The type of the output. Either `'hard'` or `'soft'`. Default is `'soft'`.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits) or `soft` (L-values).
    """

    code: TerminatedConvolutionalCode
    output_type: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError(
                "BCJR algorithm not implemented for 'tail-biting'"
            )
        if self.output_type not in ["hard", "soft"]:
            raise ValueError("'output_type' must be 'hard' or 'soft'")
        n = self.code.convolutional_code.num_output_bits
        k = self.code.convolutional_code.num_input_bits
        self._fsm = self.code.convolutional_code.finite_state_machine()
        num_states = self._fsm.num_states
        self._initial_state_distribution, self._final_state_distribution = (
            self.code.strategy.bcjr_initial_final_distributions(num_states)
        )
        self._post_process_output = self.code.strategy.bcjr_post_process_output
        self._cache_polar = (-1) ** int_to_bits(range(2**n), width=n)
        self._labeling = NaturalLabeling(k)

    def _metric_function(self, y: int, z: float) -> float:
        return 0.5 * np.dot(self._cache_polar[y], z)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.TerminatedConvolutionalCode(
            ...     convolutional_code=komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            ...     num_blocks=3,
            ...     mode="zero-termination",
            ... )

            >>> decoder = komm.BCJRDecoder(code)
            >>> decoder.decode([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6])
            array([-0.47774884, -0.61545527,  1.03018771])

            >>> decoder = komm.BCJRDecoder(code, output_type="hard")
            >>> decoder.decode([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6])
            array([1, 1, 0])
        """
        n = self.code.convolutional_code.num_output_bits

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "BCJR"))
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            symbol_posteriors = self._fsm.forward_backward(
                observed=li.reshape(-1, n),
                metric_function=self._metric_function,
                initial_state_distribution=self._initial_state_distribution,
                final_state_distribution=self._final_state_distribution,
            )
            symbol_posteriors = self._post_process_output(symbol_posteriors)
            lo = self._labeling.marginalize(symbol_posteriors).reshape(-1)
            return lo

        output = decode(input)
        if self.output_type == "hard":
            return (output < 0.0).astype(int)
        else:  # if self.output_type == "soft":
            return output
