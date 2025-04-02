from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from typing import SupportsInt

import numpy as np
import numpy.typing as npt

from .._algebra import BinaryPolynomial
from .._util.decorators import blockwise, vectorize
from . import base


class CyclicCode(base.BlockCode):
    r"""
    General binary cyclic code. A cyclic code is a [linear block code](/ref/BlockCode) such that, if $c$ is a codeword, then every cyclic shift of $c$ is also a codeword. It is characterized by its *generator polynomial* $g(X)$, of degree $m$ (the redundancy of the code), and by its *check polynomial* $h(X)$, of degree $k$ (the dimension of the code). Those polynomials are related by $g(X) h(X) = X^n + 1$, where $n = k + m$ is the length of the code.

    Examples of generator polynomials can be found in the table below.

    | Code $(n, k, d)$  | Generator polynomial $g(X)$              | Integer representation           |
    | ----------------- | ---------------------------------------- | -------------------------------- |
    | Hamming $(7,4,3)$ | $X^3 + X + 1$                            | `0b1011 = 0o13 = 11`             |
    | Simplex $(7,3,4)$ | $X^4 + X^2 + X +   1$                    | `0b10111 = 0o27 = 23`            |
    | BCH $(15,5,7)$    | $X^{10} + X^8 + X^5 + X^4 + X^2 + X + 1$ | `0b10100110111 = 0o2467 = 1335`  |
    | Golay $(23,12,7)$ | $X^{11} + X^9 + X^7 + X^6 + X^5 + X + 1$ | `0b101011100011 = 0o5343 = 2787` |

    For more details, see <cite>LC04, Ch. 5</cite>.

    The constructor expects either the generator polynomial or the check polynomial.

    Attributes:
        length: The length $n$ of the code.

        generator_polynomial: The generator polynomial $g(X)$ of the code, of degree $m$ (the redundancy of the code), specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former.

        check_polynomial: The check polynomial $h(X)$ of the code, of degree $k$ (the dimension of the code), specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former.

        systematic: Whether the encoder is systematic. Default is `True`.

    Examples:
        >>> code = komm.CyclicCode(length=23, generator_polynomial=0b101011100011)  # Golay (23, 12)
        >>> (code.length, code.dimension, code.redundancy)
        (23, 12, 11)
        >>> code.minimum_distance()
        7

        >>> code = komm.CyclicCode(length=23, check_polynomial=0b1010010011111)  # Golay (23, 12)
        >>> (code.length, code.dimension, code.redundancy)
        (23, 12, 11)
        >>> code.minimum_distance()
        7
    """

    def __init__(
        self,
        length: int,
        generator_polynomial: SupportsInt | None = None,
        check_polynomial: SupportsInt | None = None,
        systematic: bool = True,
    ):
        self._length = length
        if generator_polynomial is None and check_polynomial is None:
            raise ValueError(
                "either 'generator_polynomial' or 'check_polynomial' must be provided"
            )
        if generator_polynomial is not None and check_polynomial is None:
            self._generator_polynomial = BinaryPolynomial(generator_polynomial)
            quotient, remainder = divmod(self.modulus, self.generator_polynomial)
            if remainder != 0b0:
                raise ValueError("'generator_polynomial' must be a factor of X^n + 1")
            self._check_polynomial = quotient
            self._constructed_from = "generator_polynomial"
        elif generator_polynomial is None and check_polynomial is not None:
            self._check_polynomial = BinaryPolynomial(check_polynomial)
            quotient, remainder = divmod(self.modulus, self.check_polynomial)
            if remainder != 0b0:
                raise ValueError("'check_polynomial' must be a factor of X^n + 1")
            self._generator_polynomial = quotient
            self._constructed_from = "check_polynomial"
        self.systematic = systematic

    @cached_property
    def _encoding_strategy(self) -> "EncodingStrategy":
        if self.systematic:
            return SystematicStrategy(self)
        else:
            return NonSystematicStrategy(self)

    def __repr__(self) -> str:
        args = f"length={self.length}"
        if self._constructed_from == "generator_polynomial":
            args += f", generator_polynomial={self.generator_polynomial}"
        else:  # self._constructed_from == "check_polynomial"
            args += f", check_polynomial={self.check_polynomial}"
        args += f", systematic={self.systematic}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def length(self) -> int:
        return self._length

    @cached_property
    def dimension(self) -> int:
        return self.check_polynomial.degree

    @cached_property
    def redundancy(self) -> int:
        return self.generator_polynomial.degree

    @cached_property
    def rate(self) -> float:
        return super().rate

    @cached_property
    def generator_polynomial(self) -> BinaryPolynomial:
        return self._generator_polynomial

    @cached_property
    def check_polynomial(self) -> BinaryPolynomial:
        return self._check_polynomial

    @cached_property
    def modulus(self) -> BinaryPolynomial:
        return BinaryPolynomial.from_exponents([0, self.length])

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        return self._encoding_strategy.generator_matrix()

    @cached_property
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        raise NotImplementedError

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        return self._encoding_strategy.check_matrix()

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.dimension)
        @vectorize
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            u_poly = BinaryPolynomial.from_coefficients(u)
            v_poly = self._encoding_strategy.encode(u_poly)
            v = v_poly.coefficients(width=self.length)
            return v

        return encode(input)

    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        @vectorize
        def project(v: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            v_poly = BinaryPolynomial.from_coefficients(v)
            u_poly = self._encoding_strategy.project_word(v_poly)
            u = u_poly.coefficients(width=self.dimension)
            return u

        return project(input)

    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        return super().inverse_encode(input)

    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        @vectorize
        def check(r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            r_poly = BinaryPolynomial.from_coefficients(r)
            s_poly = r_poly % self.generator_polynomial
            s = s_poly.coefficients(width=self.redundancy)
            return s

        return check(input)

    @cache
    def codewords(self) -> npt.NDArray[np.integer]:
        return super().codewords()

    @cache
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().codeword_weight_distribution()

    @cache
    def minimum_distance(self) -> int:
        return super().minimum_distance()

    @cache
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        return super().coset_leaders()

    @cache
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        return super().coset_leader_weight_distribution()

    @cache
    def packing_radius(self) -> int:
        return super().packing_radius()

    @cache
    def covering_radius(self) -> int:
        return super().covering_radius()


@dataclass
class EncodingStrategy(ABC):
    code: CyclicCode

    @abstractmethod
    def generator_matrix(self) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def check_matrix(self) -> npt.NDArray[np.integer]: ...

    @abstractmethod
    def encode(self, u_poly: BinaryPolynomial) -> BinaryPolynomial: ...

    @abstractmethod
    def project_word(self, v_poly: BinaryPolynomial) -> BinaryPolynomial: ...


class SystematicStrategy(EncodingStrategy):
    # See [McE04, Sec. 8.1].
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        n, k, m = self.code.length, self.code.dimension, self.code.redundancy
        generator_matrix = np.empty((k, n), dtype=int)
        X = BinaryPolynomial(0b10)  # The polynomial X
        for i in range(k):
            row_poly = X ** (m + i) + X ** (m + i) % self.code.generator_polynomial
            generator_matrix[i] = row_poly.coefficients(width=n)
        return generator_matrix

    def check_matrix(self) -> npt.NDArray[np.integer]:
        n, m = self.code.length, self.code.redundancy
        check_matrix = np.empty((m, n), dtype=int)
        X = BinaryPolynomial(0b10)
        for j in range(n):
            col_poly = X**j % self.code.generator_polynomial
            check_matrix[:, j] = col_poly.coefficients(width=m)
        return check_matrix

    def encode(self, u_poly: BinaryPolynomial) -> BinaryPolynomial:
        u_poly_shifted = u_poly << self.code.redundancy
        b_poly = u_poly_shifted % self.code.generator_polynomial
        v_poly = u_poly_shifted + b_poly
        return v_poly

    def project_word(self, v_poly: BinaryPolynomial) -> BinaryPolynomial:
        u_poly = v_poly >> self.code.redundancy
        return u_poly


class NonSystematicStrategy(EncodingStrategy):
    # See [McE04, Sec. 8.1].
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        n, k = self.code.length, self.code.dimension
        generator_matrix = np.empty((k, n), dtype=int)
        X = BinaryPolynomial(0b10)  # The polynomial X
        for i in range(k):
            row_poly = X**i * self.code.generator_polynomial
            generator_matrix[i] = row_poly.coefficients(width=n)
        return generator_matrix

    def check_matrix(self) -> npt.NDArray[np.integer]:
        n, m = self.code.length, self.code.redundancy
        check_matrix = np.empty((m, n), dtype=int)
        X = BinaryPolynomial(0b10)
        for i in range(m):
            row_poly = X**i * self.code.check_polynomial.reciprocal()
            check_matrix[i] = row_poly.coefficients(width=n)
        return check_matrix

    def encode(self, u_poly: BinaryPolynomial) -> BinaryPolynomial:
        v_poly = u_poly * self.code.generator_polynomial
        return v_poly

    def project_word(self, v_poly: BinaryPolynomial) -> BinaryPolynomial:
        u_poly = v_poly // self.code.generator_polynomial
        return u_poly
