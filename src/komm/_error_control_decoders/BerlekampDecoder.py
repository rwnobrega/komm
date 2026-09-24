from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._algebra import bifield
from .._algebra.FiniteBifield import FiniteBifield
from .._error_control_block.BCHCode import BCHCode
from .._error_control_block.ReedSolomonCode import ReedSolomonCode
from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.decorators import blockwise
from .util import get_pbar


@dataclass
class BerlekampDecoder(abc.CodewordDecoder[BCHCode | ReedSolomonCode]):
    r"""
    Berlekamp decoder for [BCH codes](/ref/BCHCode) and [Reed–Solomon codes](/ref/ReedSolomonCode). For more details, see <cite>LC04, Sec. 6.3</cite> and <cite>LC04, Sec. 7.4</cite>.

    Parameters:
        code: The BCH or Reed–Solomon code to be used for decoding.

    Notes:
        - Input type: `hard` (bits).
        - Output type: `hard` (bits).
    """

    code: BCHCode | ReedSolomonCode

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode_to_codeword(
            ...     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ... )
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> code = komm.ReedSolomonCode(3, 5)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode_to_codeword(
            ...     [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            ... )
            array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
        """
        field, alpha = self.code.field, int(self.code.alpha)
        n = field.order - 1  # Length in symbols.
        width = self.code.mu if isinstance(self.code, ReedSolomonCode) else 1
        points = bifield.power(field, alpha, np.arange(1, self.code.delta))
        inverses = bifield.power(field, alpha, -np.arange(n))

        @blockwise(self.code.length)
        def decode_to_codeword(r: npt.NDArray[np.integer]):
            r = bits_to_int(r, width=width)
            # See [LC04, Sec. 6.2].
            syndromes = bifield.horner(field, r, points)
            v_hat = r.copy()
            pbar = get_pbar(np.size(input) // self.code.length, "Berlekamp")
            for i in np.ndindex(r.shape[:-1]):
                pbar.update()
                if not syndromes[i].any():
                    continue
                sigma = berlekamp_algorithm(field, syndromes[i])
                # Chien search.
                e_loc = np.flatnonzero(bifield.horner(field, sigma, inverses) == 0)
                if len(e_loc) != len(sigma) - 1:
                    continue
                # Error values
                if isinstance(self.code, BCHCode):
                    e_val = 1
                else:
                    roots = inverses[e_loc]
                    e_val = forney_algorithm(field, syndromes[i], sigma, roots)
                v_hat[i][e_loc] ^= e_val
            return int_to_bits(v_hat, width=width)

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode(
            ...     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ... )
            array([0, 0, 0, 0, 0])

            >>> code = komm.ReedSolomonCode(3, 5)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode(
            ...     [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            ... )
            array([0, 0, 0, 1, 1, 0, 1, 0, 0])
        """
        return self.code.project_word(self.decode_to_codeword(input))


def berlekamp_algorithm(
    field: FiniteBifield, syndrome: npt.NDArray[np.integer]
) -> npt.NDArray[np.integer]:
    # Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.
    # See [LC04, Sec. 6.3].
    sigma = {-1: np.array([1]), 0: np.array([1])}
    discrepancy = {-1: 1}
    degree = {-1: 0, 0: 0}

    # In [LC04]: μ <-> j and ρ <-> k.
    for j in range(len(syndrome)):
        products = bifield.multiply(field, sigma[j], syndrome[j::-1][: len(sigma[j])])
        discrepancy[j] = np.bitwise_xor.reduce(products)
        if discrepancy[j] == 0:
            degree[j + 1] = degree[j]
            sigma[j + 1] = sigma[j]
            continue
        candidates = [i for i in range(-1, j) if discrepancy[i] != 0]
        k = max(candidates, key=lambda i: i - degree[i])
        degree[j + 1] = max(degree[j], degree[k] + j - k)
        # See [LC04, eq. (6.25)].
        ratio = bifield.divide(field, discrepancy[j], discrepancy[k])
        correction = bifield.multiply(field, sigma[k], ratio)
        sigma[j + 1] = np.zeros(degree[j + 1] + 1, dtype=int)
        sigma[j + 1][: len(sigma[j])] = sigma[j]
        sigma[j + 1][j - k : j - k + len(correction)] ^= correction

    return sigma[len(syndrome)]


def forney_algorithm(
    field: FiniteBifield,
    syndrome: npt.NDArray[np.integer],
    sigma: npt.NDArray[np.integer],
    roots: npt.NDArray[np.integer],
) -> npt.NDArray[np.integer]:
    # Forney's formula, see [McE04, eq. (9.72)].
    # E_i = ω(α^-i) / σ'(α^-i).
    # Key equation: ω(x) = σ(x) S(x) mod x^m.
    omega = bifield.convolve(field, syndrome, sigma)[: len(syndrome)]
    sigma_prime = sigma[1:].copy()
    sigma_prime[1::2] = 0  # In σ'(x), only odd powers survive.
    numerator = bifield.horner(field, omega, roots)
    denominator = bifield.horner(field, sigma_prime, roots)
    return bifield.divide(field, numerator, denominator)
