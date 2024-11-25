import numpy as np
import numpy.typing as npt

from ..ReedMullerCode import ReedMullerCode
from ..registry import RegistryBlockDecoder


def decode_reed(
    code: ReedMullerCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    # See [LC04, p. 105–114, 439–440].
    u_hat = np.empty(code.dimension, dtype=int)
    bx = np.copy(r)
    for i, partition in enumerate(code.reed_partitions()):
        checksums = np.count_nonzero(bx[partition], axis=1) % 2
        u_hat[i] = np.count_nonzero(checksums) > len(checksums) // 2
        bx ^= u_hat[i] * code.generator_matrix[i]
    return u_hat


RegistryBlockDecoder.register(
    "reed",
    {
        "description": (
            "Reed decoding algorithm for Reed–Muller codes. It's a majority-logic"
            " decoding algorithm."
        ),
        "decoder": decode_reed,
        "type_in": "hard",
        "type_out": "hard",
        "target": "message",
    },
)


def decode_weighted_reed(
    code: ReedMullerCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    # See [LC04, p. 440–442].
    r = np.asarray(r)
    u_hat = np.empty(code.dimension, dtype=int)
    bx = (r < 0).astype(int)
    for i, partition in enumerate(code.reed_partitions()):
        checksums = np.count_nonzero(bx[partition], axis=1) % 2
        min_reliability = np.min(np.abs(r[partition]), axis=1)
        decision_var = np.dot(1 - 2 * checksums, min_reliability)
        u_hat[i] = decision_var < 0
        bx ^= u_hat[i] * code.generator_matrix[i]
    return u_hat


RegistryBlockDecoder.register(
    "weighted-reed",
    {
        "description": "Weighted Reed decoding algorithm for Reed–Muller codes.",
        "decoder": decode_weighted_reed,
        "type_in": "soft",
        "type_out": "hard",
        "target": "message",
    },
)
