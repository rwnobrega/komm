from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .._algebra.BinaryPolynomial import BinaryPolynomial
from .._algebra.FiniteBifield import FiniteBifield, FiniteBifieldElement, find_roots
from .._error_control_block import BCHCode
from .._util.decorators import blockwise, vectorize, with_pbar
from . import base
from .util import get_pbar


@dataclass
class BerlekampDecoder(base.BlockDecoder[BCHCode]):
    r"""
    Berlekamp decoder for [BCH codes](/ref/BCHCode). For more details, see <cite>LC04, Sec. 6.3</cite>.

    Parameters:
        code: The BCH code to be used for decoding.

    Notes:
        - Input type: `hard`.
        - Output type: `hard`.
    """

    code: BCHCode

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            array([0, 0, 0, 0, 0])
        """

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Berlekamp"))
        def decode(r: npt.NDArray[np.integer]):
            r_poly = BinaryPolynomial.from_coefficients(r)
            syndrome = self.code.bch_syndrome(r_poly)
            if all(x == self.code.field.zero for x in syndrome):
                return self.code.project_word(r)
            sigma_poly = berlekamp_algorithm(self.code, syndrome)
            roots = find_roots(self.code.field, sigma_poly)
            e_loc = [e.inverse().logarithm(self.code.alpha) for e in roots]
            e_hat = np.bincount(e_loc, minlength=self.code.length)
            v_hat = (r + e_hat) % 2
            u_hat = self.code.project_word(v_hat)
            return u_hat

        return decode(input)


def berlekamp_algorithm(
    code: BCHCode, syndrome: list[Any]
) -> list[FiniteBifieldElement[FiniteBifield]]:
    # Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.
    # See [LC04, Sec. 6.3].
    field = code.field
    delta = code.delta
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
