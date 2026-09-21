from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import abc
from .._algebra.FiniteBifield import (
    FiniteBifield,
    FiniteBifieldElement,
    find_roots,
    horner,
)
from .._error_control_block.BCHCode import BCHCode
from .._util.decorators import blockwise
from .util import get_pbar


@dataclass
class BerlekampDecoder(abc.CodewordDecoder[BCHCode]):
    r"""
    Berlekamp decoder for [BCH codes](/ref/BCHCode). For more details, see <cite>LC04, Sec. 6.3</cite>.

    Parameters:
        code: The BCH code to be used for decoding.

    Notes:
        - Input type: `hard` (bits).
        - Output type: `hard` (bits).
    """

    code: BCHCode

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode_to_codeword([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        field = self.code.field
        points = field.power(int(self.code.alpha), np.arange(1, self.code.delta))

        @blockwise(self.code.length)
        def decode_to_codeword(r: npt.NDArray[np.integer]):
            # See [LC04, Sec. 6.2].
            syndromes = horner(field, r, points)
            v_hat = r.copy()
            pbar = get_pbar(np.size(input) // self.code.length, "Berlekamp")
            for i in np.ndindex(r.shape[:-1]):
                pbar.update()
                if not syndromes[i].any():
                    continue
                syndrome = [field(s) for s in syndromes[i]]
                sigma_poly = berlekamp_algorithm(self.code, syndrome)
                roots = find_roots(field, sigma_poly)
                if len(roots) != len(sigma_poly) - 1:
                    continue
                e_loc = [e.inverse().logarithm(self.code.alpha) for e in roots]
                e_hat = np.bincount(e_loc, minlength=self.code.length)
                v_hat[i] = (r[i] + e_hat) % 2
            return v_hat

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder.decode([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            array([0, 0, 0, 0, 0])
        """
        return self.code.project_word(self.decode_to_codeword(input))


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
            # See [LC04, eq. (6.25)].
            sigma[j + 1] = [
                fst[i] + snd[i] * discrepancy[j] / discrepancy[k]
                for i in range(degree[j + 1] + 1)
            ]
        if j < delta - 2:
            discrepancy[j + 1] = syndrome[j + 1]
            for i in range(degree[j + 1]):
                discrepancy[j + 1] += sigma[j + 1][i + 1] * syndrome[j - i]

    return sigma[delta - 1]
