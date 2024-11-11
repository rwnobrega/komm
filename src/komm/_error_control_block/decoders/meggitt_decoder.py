import numpy as np
import numpy.typing as npt

from ..._algebra import BinaryPolynomial
from .._registry import RegistryBlockDecoder
from ..CyclicCode import CyclicCode


def decode_meggitt(code: CyclicCode, r: npt.ArrayLike) -> np.ndarray:
    # See [XiD03, Sec. 3.4] for more details.
    meggitt_table = code.meggitt_table
    r_poly = BinaryPolynomial.from_coefficients(r)
    s_poly = r_poly % code.generator_polynomial
    if s_poly == 0:
        return np.asarray(r)
    e_poly_hat = BinaryPolynomial(0)
    for j in range(code.length):
        if s_poly in meggitt_table:
            e_poly_hat = meggitt_table[s_poly] // BinaryPolynomial(1 << j)
            break
        s_poly = (s_poly << 1) % code.generator_polynomial
    return (r_poly + e_poly_hat).coefficients(code.length)


RegistryBlockDecoder.register(
    "meggitt",
    {
        "description": "Meggitt decoder",
        "decoder": decode_meggitt,
        "type_in": "hard",
        "type_out": "hard",
        "target": "codeword",
    },
)
