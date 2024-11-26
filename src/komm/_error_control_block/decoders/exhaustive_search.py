import numpy as np
import numpy.typing as npt

from ..BlockCode import BlockCode
from ..registry import RegistryBlockDecoder


def decode_exhaustive_search_hard(
    code: BlockCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    codewords = code.codewords()
    metrics = np.count_nonzero(r != codewords, axis=1)
    v_hat = codewords[np.argmin(metrics)]
    return v_hat


RegistryBlockDecoder.register(
    "exhaustive-search-hard",
    {
        "description": (
            "Exhaustive search (hard-decision). Minimum Hamming distance decoder"
        ),
        "decoder": decode_exhaustive_search_hard,
        "type_in": "hard",
        "type_out": "hard",
        "target": "codeword",
    },
)


def decode_exhaustive_search_soft(
    code: BlockCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    codewords = code.codewords()
    metrics = np.dot(codewords, r)
    v_hat = codewords[np.argmin(metrics)]
    return v_hat


RegistryBlockDecoder.register(
    "exhaustive-search-soft",
    {
        "description": (
            "Exhaustive search (soft-decision). Minimum Euclidean distance decoder"
        ),
        "decoder": decode_exhaustive_search_soft,
        "type_in": "soft",
        "type_out": "hard",
        "target": "codeword",
    },
)
