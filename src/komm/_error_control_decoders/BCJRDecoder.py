from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt

from .._error_control_convolutional import TerminatedConvolutionalCode
from .._modulation.labelings import labeling_natural
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.information_theory import marginalize_bits
from . import base
from .util import get_pbar


@dataclass
class BCJRDecoder(base.BlockDecoder[TerminatedConvolutionalCode]):
    r"""
    Bahl–Cocke–Jelinek–Raviv (BCJR) decoder for [terminated convolutional codes](/ref/TerminatedConvolutionalCode). For more details, see <cite>LC04, Sec. 12.6</cite>.

    Parameters:
        code: The terminated convolutional code to be used for decoding.
        snr: The signal-to-noise ratio (SNR) of the channel (linear, not decibel).

    Notes:
        - Input type: `soft`.
        - Output type: `soft`.
    """

    code: TerminatedConvolutionalCode
    snr: float = 1.0

    def __post_init__(self) -> None:
        if self.code.mode == "tail-biting":
            raise NotImplementedError(
                "BCJR algorithm not implemented for 'tail-biting'"
            )
        self._fsm = self.code.convolutional_code.finite_state_machine()
        num_states = self._fsm.num_states
        self._initial_state_distribution, self._final_state_distribution = (
            self.code.strategy.bcjr_initial_final_distributions(num_states)
        )
        self._post_process_output = self.code.strategy.bcjr_post_process_output
        n = self.code.convolutional_code.num_output_bits
        self._cache_polar = (-1) ** int_to_bits(range(2**n), width=n)
        self._labeling = labeling_natural(n)

    def _metric_function(self, y: int, z: float) -> float:
        return 2.0 * self.snr * np.dot(self._cache_polar[y], z)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b1]], feedback_polynomials=[0b11])
            >>> code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode="zero-termination")
            >>> decoder = komm.BCJRDecoder(code, snr=0.25)
            >>> decoder([-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6])
            array([-0.47774884, -0.61545527,  1.03018771])
        """
        n = self.code.convolutional_code.num_output_bits

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "BCJR"))
        def decode(r: npt.NDArray[np.floating]):
            symbol_posteriors = self._fsm.forward_backward(
                observed_sequence=r.reshape(-1, n),
                metric_function=self._metric_function,
                initial_state_distribution=self._initial_state_distribution,
                final_state_distribution=self._final_state_distribution,
            )
            symbol_posteriors = self._post_process_output(symbol_posteriors)
            soft_bits = np.apply_along_axis(
                func1d=partial(marginalize_bits, labeling=self._labeling),
                axis=1,
                arr=symbol_posteriors,
            ).ravel()
            return soft_bits

        return decode(input)
