import numpy as np
import numpy.typing as npt

from .._registry import RegistryBlockDecoder
from ..RepetitionCode import RepetitionCode


def decode_majority_logic_repetition_code(
    code: RepetitionCode, r: npt.ArrayLike
) -> np.ndarray:
    u_hat = np.array([np.argmax(np.bincount(r))])
    return u_hat


RegistryBlockDecoder.register(
    "majority_logic_repetition_code",
    {
        "description": "Majority-logic decoder. A hard-decision decoder for Repetition codes only.",
        "decoder": decode_majority_logic_repetition_code,
        "type_in": "hard",
        "type_out": "hard",
        "target": "message",
    },
)
