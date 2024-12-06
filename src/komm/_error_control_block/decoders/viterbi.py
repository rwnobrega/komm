from typing import Any

import numpy as np
import numpy.typing as npt

from ..._error_control_convolutional.TerminatedConvolutionalCode import (
    TerminatedConvolutionalCode,
)
from ..._finite_state_machine.FiniteStateMachine import MetricFunction
from ..._util import int_to_bits
from ..registry import RegistryBlockDecoder


def decode_viterbi(
    code: TerminatedConvolutionalCode,
    r: npt.ArrayLike,
    metric_function: MetricFunction[Any],
) -> npt.NDArray[np.int_]:
    if code.mode == "tail-biting":
        raise NotImplementedError("Viterbi algorithm not implemented for 'tail-biting'")

    k0 = code.convolutional_code.num_input_bits
    n0 = code.convolutional_code.num_output_bits
    mu = code.convolutional_code.memory_order
    fsm = code.convolutional_code.finite_state_machine()

    initial_metrics = np.full(fsm.num_states, fill_value=np.inf)
    initial_metrics[0] = 0.0

    xs_hat, final_metrics = fsm.viterbi(
        observed_sequence=np.reshape(r, shape=(-1, n0)),
        metric_function=metric_function,
        initial_metrics=initial_metrics,
    )

    if code.mode == "direct-truncation":
        s_hat = np.argmin(final_metrics)
        x_hat = xs_hat[:, s_hat]
    else:  # code.mode == "zero-termination"
        x_hat = xs_hat[:, 0][:-mu]

    u_hat = int_to_bits(x_hat, width=k0).ravel()
    return u_hat


def decode_viterbi_hard(
    code: TerminatedConvolutionalCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    def metric_function(y: int, z: float) -> float:
        return np.count_nonzero(code.cache_bit[y] != z)

    return decode_viterbi(code, r, metric_function)


RegistryBlockDecoder.register(
    "viterbi-hard",
    {
        "description": "Viterbi (hard-decision)",
        "decoder": decode_viterbi_hard,
        "type_in": "hard",
        "type_out": "hard",
        "target": "message",
    },
)


def decode_viterbi_soft(
    code: TerminatedConvolutionalCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    def metric_function(y: int, z: int) -> float:
        return np.dot(code.cache_bit[y], z)

    return decode_viterbi(code, r, metric_function)


RegistryBlockDecoder.register(
    "viterbi-soft",
    {
        "description": "Viterbi (soft-decision)",
        "decoder": decode_viterbi_soft,
        "type_in": "soft",
        "type_out": "hard",
        "target": "message",
    },
)
