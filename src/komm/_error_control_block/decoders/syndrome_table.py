import numpy as np
import numpy.typing as npt

from ..._util import binlist2int
from .._registry import RegistryBlockDecoder
from ..BlockCode import BlockCode


def decode_syndrome_table(code: BlockCode, r: npt.ArrayLike) -> np.ndarray:
    coset_leaders = code.coset_leaders
    s = np.dot(r, code.check_matrix.T) % 2
    e_hat = coset_leaders[binlist2int(s)]
    v_hat = np.bitwise_xor(r, e_hat)
    return v_hat


RegistryBlockDecoder.register(
    "syndrome_table",
    {
        "description": "Syndrome table decoder",
        "decoder": decode_syndrome_table,
        "type_in": "hard",
        "type_out": "hard",
        "target": "codeword",
    },
)
