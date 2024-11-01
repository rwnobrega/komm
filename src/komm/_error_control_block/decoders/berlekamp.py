import numpy as np
import numpy.typing as npt

from ..._algebra import BinaryPolynomial, FiniteBifield
from .._registry import RegistryBlockDecoder
from ..BCHCode import BCHCode

# def bch_general_decoder(field: FiniteBifield, r, syndrome_computer, key_equation_solver, root_finder):
#     # General BCH decoder. See [LC04, p. 205–209].
#     r_poly = BinaryPolynomial.from_coefficients(r)
#     s_poly = syndrome_computer(r_poly)
#     if np.all([x == field(0) for x in s_poly]):
#         return r
#     error_location_polynomial = key_equation_solver(s_poly)
#     error_locations = [e.inverse().logarithm() for e in root_finder(error_location_polynomial)]
#     e_hat = np.bincount(error_locations, minlength=r.size)
#     v_hat = np.bitwise_xor(r, e_hat)
#     return v_hat


def bch_syndrome_vector(code: BCHCode, r_poly: BinaryPolynomial):
    # BCH syndrome computation. See [LC04, p. 205–209].
    alpha = code.field.primitive_element
    s_vec = np.array(
        [(r_poly % code.phi(i)).evaluate(alpha**i) for i in range(1, code.delta)],
        dtype=object,
    )
    return s_vec


def find_roots(field: FiniteBifield, coefficients) -> list[FiniteBifield._Element]:
    # Exhaustive search.
    roots = []
    for i in range(field.order):
        x = field(i)
        evaluated = field(0)
        for coefficient in reversed(coefficients):  # Horner's method
            evaluated = evaluated * x + coefficient
        if evaluated == field(0):
            roots.append(x)
            if len(roots) >= len(coefficients) - 1:
                break
    return roots


def berlekamp_algorithm(field: FiniteBifield, delta: int, syndrome_polynomial):
    # Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.
    # See [LC04, p. 209–212] and [RL09, p. 114–121].
    sigma = {
        -1: np.array([field(1)], dtype=object),
        0: np.array([field(1)], dtype=object),
    }
    discrepancy = {-1: field(1), 0: syndrome_polynomial[0]}
    degree = {-1: 0, 0: 0}

    for j in range(delta - 1):
        if discrepancy[j] == field(0):
            degree[j + 1] = degree[j]
            sigma[j + 1] = sigma[j]
        else:
            rho, max_so_far = -1, -1
            for i in range(-1, j):
                if discrepancy[i] != field(0) and i - degree[i] > max_so_far:
                    rho, max_so_far = i, i - degree[i]
            degree[j + 1] = max(degree[j], degree[rho] + j - rho)
            sigma[j + 1] = np.array([field(0)] * (degree[j + 1] + 1), dtype=object)
            first_guy = np.array([field(0)] * (degree[j + 1] + 1), dtype=object)
            first_guy[: degree[j] + 1] = sigma[j]
            second_guy = np.array([field(0)] * (degree[j + 1] + 1), dtype=object)
            second_guy[j - rho : degree[rho] + j - rho + 1] = sigma[rho]
            e = discrepancy[j] / discrepancy[rho]
            second_guy = np.array([e * x for x in second_guy], dtype=object)
            sigma[j + 1] = first_guy + second_guy
        if j < delta - 2:
            discrepancy[j + 1] = syndrome_polynomial[j + 1]
            for idx in range(1, degree[j + 1] + 1):
                discrepancy[j + 1] += (
                    sigma[j + 1][idx] * syndrome_polynomial[j + 1 - idx]
                )

    return sigma[delta - 1]


def decode_berlekamp(code: BCHCode, r: npt.ArrayLike) -> np.ndarray:
    r = np.asarray(r)
    r_poly = BinaryPolynomial.from_coefficients(r)
    s_poly = bch_syndrome_vector(code, r_poly)
    if np.all([x == code.field(0) for x in s_poly]):
        return r
    sigma_poly = berlekamp_algorithm(code.field, code.delta, s_poly)
    roots = find_roots(code.field, sigma_poly)
    e_loc = [e.inverse().logarithm() for e in roots]
    e_hat = np.bincount(e_loc, minlength=code.length)
    v_hat = np.bitwise_xor(r, e_hat)
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
