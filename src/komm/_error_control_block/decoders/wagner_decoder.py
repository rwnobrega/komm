import numpy as np
import numpy.typing as npt

from .._registry import RegistryBlockDecoder
from ..SingleParityCheckCode import SingleParityCheckCode


def decode_wagner(code: SingleParityCheckCode, r: npt.ArrayLike) -> np.ndarray:
    # See Costello, Forney: Channel Coding: The Road to Channel Capacity.
    r = np.asarray(r)
    v_hat = r < 0
    if np.count_nonzero(v_hat) % 2 != 0:
        i = np.argmin(np.abs(r))
        v_hat[i] ^= 1
    return v_hat


RegistryBlockDecoder.register(
    "wagner",
    {
        "description": "Wagner decoder. A soft-decision decoder for SingleParityCheck codes only.",
        "decoder": decode_wagner,
        "type_in": "soft",
        "type_out": "hard",
        "target": "codeword",
    },
)
