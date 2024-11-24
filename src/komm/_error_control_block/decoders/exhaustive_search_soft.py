import numpy as np
import numpy.typing as npt

from ..BlockCode import BlockCode
from ..registry import RegistryBlockDecoder


def decode_exhaustive_search_soft(code: BlockCode, r: npt.ArrayLike) -> np.ndarray:
    codewords = code.codewords()
    metrics = np.dot(r, codewords.T)
    v_hat = codewords[np.argmin(metrics)]
    return v_hat


RegistryBlockDecoder.register(
    "exhaustive_search_soft",
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
