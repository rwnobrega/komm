from typing import Any

import numpy as np
import numpy.typing as npt

from ..._algebra.BinaryPolynomial import BinaryPolynomial
from ..._algebra.FiniteBifield import F, FiniteBifield, FiniteBifieldElement, find_roots
from ..BCHCode import BCHCode
from ..registry import RegistryBlockDecoder


def bch_syndrome(
    code: BCHCode,
    r_poly: BinaryPolynomial,
) -> list[FiniteBifieldElement[FiniteBifield]]:
    # BCH syndrome computation. See [LC04, p. 205–209].
    alpha = code.field.primitive_element
    return [(r_poly % code.phi(i)).evaluate(alpha**i) for i in range(1, code.delta)]


def berlekamp_algorithm(
    field: F,
    delta: int,
    syndrome: list[Any],
) -> list[FiniteBifieldElement[F]]:
    # Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.
    # See [LC04, p. 209–212] and [RL09, p. 114–121].
    sigma = {-1: [field.one], 0: [field.one]}
    discrepancy = {-1: field.one, 0: syndrome[0]}
    degree = {-1: 0, 0: 0}

    # In [LC04]: μ <-> j and ρ <-> k.
    for j in range(delta - 1):
        if discrepancy[j] == field.zero:
            degree[j + 1] = degree[j]
            sigma[j + 1] = sigma[j]
        else:
            k, max_so_far = -1, -1
            for i in range(-1, j):
                if discrepancy[i] != field.zero and i - degree[i] > max_so_far:
                    k, max_so_far = i, i - degree[i]
            degree[j + 1] = max(degree[j], degree[k] + j - k)
            fst = [field.zero] * (degree[j + 1] + 1)
            fst[: degree[j] + 1] = sigma[j]
            snd = [field.zero] * (degree[j + 1] + 1)
            snd[j - k : degree[k] + j - k + 1] = sigma[k]
            # [LC04, Eq. (6.25)]
            sigma[j + 1] = [
                fst[i] + snd[i] * discrepancy[j] / discrepancy[k]
                for i in range(degree[j + 1] + 1)
            ]
        if j < delta - 2:
            discrepancy[j + 1] = syndrome[j + 1]
            for i in range(degree[j + 1]):
                discrepancy[j + 1] += sigma[j + 1][i + 1] * syndrome[j - i]

    return sigma[delta - 1]


def decode_berlekamp(
    code: BCHCode,
    r: npt.ArrayLike,
) -> npt.NDArray[np.int_]:
    alpha = code.field.primitive_element
    r = np.asarray(r)
    r_poly = BinaryPolynomial.from_coefficients(r)
    syndrome = bch_syndrome(code, r_poly)
    if all(x == code.field.zero for x in syndrome):
        return r
    sigma_poly = berlekamp_algorithm(code.field, code.delta, syndrome)
    roots = find_roots(code.field, sigma_poly)
    e_loc = [e.inverse().logarithm(alpha) for e in roots]
    e_hat = np.bincount(e_loc, minlength=code.length)
    v_hat = (r + e_hat) % 2
    return v_hat


RegistryBlockDecoder.register(
    "berlekamp",
    {
        "description": "Berlekamp decoder",
        "decoder": decode_berlekamp,
        "type_in": "hard",
        "type_out": "hard",
        "target": "codeword",
    },
)
