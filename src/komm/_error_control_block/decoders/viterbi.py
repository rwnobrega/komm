import numpy as np
import numpy.typing as npt

from ..._util import unpack
from .._registry import RegistryBlockDecoder
from ..TerminatedConvolutionalCode import TerminatedConvolutionalCode


def decode_viterbi(
    code: TerminatedConvolutionalCode, r: npt.ArrayLike, metric_function
) -> np.ndarray:
    if code.mode == "tail-biting":
        raise NotImplementedError("Viterbi algorithm not implemented for 'tail-biting'")

    k0, n0, mu, fsm = (
        code.convolutional_code.num_input_bits,
        code.convolutional_code.num_output_bits,
        code.convolutional_code.memory_order,
        code.convolutional_code.finite_state_machine,
    )

    initial_metrics = np.full(fsm.num_states, fill_value=np.inf)
    initial_metrics[0] = 0.0

    z = np.reshape(r, shape=(-1, n0))
    xs_hat, final_metrics = fsm.viterbi(
        z, metric_function=metric_function, initial_metrics=initial_metrics
    )

    if code.mode == "direct-truncation":
        s_hat = np.argmin(final_metrics)
        x_hat = xs_hat[:, s_hat]
    else:  # code.mode == "zero-termination"
        x_hat = xs_hat[:, 0][:-mu]

    u_hat = unpack(x_hat, width=k0)
    return u_hat


def decode_viterbi_hard(
    code: TerminatedConvolutionalCode, r: npt.ArrayLike
) -> np.ndarray:
    metric_function = lambda y, z: np.count_nonzero(code.cache_bit[y] != z)
    return decode_viterbi(code, r, metric_function)


RegistryBlockDecoder.register(
    "viterbi_hard",
    {
        "description": "Viterbi (hard-decision)",
        "decoder": decode_viterbi_hard,
        "type_in": "hard",
        "type_out": "hard",
        "target": "message",
    },
)


def decode_viterbi_soft(
    code: TerminatedConvolutionalCode, r: npt.ArrayLike
) -> np.ndarray:
    metric_function = lambda y, z: np.dot(code.cache_bit[y], z)
    return decode_viterbi(code, r, metric_function)


RegistryBlockDecoder.register(
    "viterbi_soft",
    {
        "description": "Viterbi (soft-decision)",
        "decoder": decode_viterbi_soft,
        "type_in": "soft",
        "type_out": "hard",
        "target": "message",
    },
)
