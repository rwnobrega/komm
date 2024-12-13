from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from komm._algebra import FiniteBifield

from .. import abc
from .._algebra.BinaryPolynomial import BinaryPolynomial
from .._algebra.FiniteBifield import FiniteBifieldElement, find_roots
from .._error_control_block import BCHCode
from .._util.decorators import vectorized_method


@dataclass
class BerlekampDecoder(abc.BlockDecoder[BCHCode]):
    r"""
    Berlekamp decoder for [BCH codes](/ref/BCHCode). For more details, see <cite>LC04, Sec. 6.3</cite>.

    Parameters:
        code: The BCH code to be used for decoding.

    Notes:
        - Input type: `hard`.
        - Output type: `hard`.

    :::komm.BerlekampDecoder.BerlekampDecoder._decode
    """

    code: BCHCode

    def __post_init__(self) -> None:
        self._alpha = self.code.field.primitive_element

    def _berlekamp_algorithm(
        self, syndrome: list[Any]
    ) -> list[FiniteBifieldElement[FiniteBifield]]:
        # Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.
        # See [LC04, Sec. 6.3].
        field = self.code.field
        delta = self.code.delta
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

    @vectorized_method
    def _decode(self, input: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        r"""
        Parameters: Input:
            input: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

        Returns: Output:
            output: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

        Examples:
            >>> code = komm.BCHCode(4, 7)
            >>> decoder = komm.BerlekampDecoder(code)
            >>> decoder([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            array([0, 0, 0, 0, 0])
        """
        r_poly = BinaryPolynomial.from_coefficients(input)
        syndrome = self.code.bch_syndrome(r_poly)
        if all(x == self.code.field.zero for x in syndrome):
            return self.code.inverse_encode(input)
        sigma_poly = self._berlekamp_algorithm(syndrome)
        roots = find_roots(self.code.field, sigma_poly)
        e_loc = [e.inverse().logarithm(self._alpha) for e in roots]
        e_hat = np.bincount(e_loc, minlength=self.code.length)
        v_hat = (input + e_hat) % 2
        output = self.code.inverse_encode(v_hat)
        return output