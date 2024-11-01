import numpy as np
import numpy.typing as npt

from .._registry import RegistryBlockDecoder
from ..TerminatedConvolutionalCode import TerminatedConvolutionalCode


def decode_bcjr(
    code: TerminatedConvolutionalCode, r: npt.ArrayLike, *, snr: float
) -> np.ndarray:
    if code.mode == "tail-biting":
        raise NotImplementedError("BCJR algorithm not implemented for 'tail-biting'")

    metric_function = lambda y, z: 2.0 * snr * np.dot(code.cache_polar[y], z)
    n0, mu, fsm = (
        code.convolutional_code.num_output_bits,
        code.convolutional_code.memory_order,
        code.convolutional_code.finite_state_machine,
    )

    if code.mode == "direct-truncation":
        initial_state_distribution = np.eye(1, fsm.num_states, 0)
        final_state_distribution = np.ones(fsm.num_states) / fsm.num_states
    else:  # code.mode == "zero-termination"
        initial_state_distribution = np.eye(1, fsm.num_states, 0)
        final_state_distribution = np.eye(1, fsm.num_states, 0)

    z = np.reshape(r, shape=(-1, n0))
    input_posteriors = fsm.forward_backward(
        z,
        metric_function=metric_function,
        initial_state_distribution=initial_state_distribution,
        final_state_distribution=final_state_distribution,
    )

    if code.mode == "zero-termination":
        input_posteriors = input_posteriors[:-mu]

    return np.log(input_posteriors[:, 0] / input_posteriors[:, 1])


RegistryBlockDecoder.register(
    "bcjr",
    {
        "description": "Bahl–Cocke–Jelinek–Raviv (BCJR)",
        "decoder": decode_bcjr,
        "type_in": "soft",
        "type_out": "soft",
        "target": "message",
    },
)
