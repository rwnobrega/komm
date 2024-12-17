from functools import cached_property

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .._algebra import BinaryPolynomial
from .._util.decorators import vectorized_method
from .BlockCode import BlockCode


@frozen(kw_only=True)
class CyclicCode(BlockCode):
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

    _length: int = field(default=None, repr=False, alias="length")
    _generator_polynomial: BinaryPolynomial | int = field(
        default=None, repr=False, alias="generator_polynomial"
    )
    _check_polynomial: BinaryPolynomial | int = field(
        default=None, repr=False, alias="check_polynomial"
    )
    _systematic: bool = field(default=True, repr=False, alias="systematic")

    def __attrs_post_init__(self) -> None:
        if (
            self._generator_polynomial is not None
            and self.modulus % self.generator_polynomial != 0b0
        ):
            raise ValueError("'generator_polynomial' must be a factor of X^n + 1")
        if (
            self._check_polynomial is not None
            and self.modulus % self.check_polynomial != 0b0
        ):
            raise ValueError("'check_polynomial' must be a factor of X^n + 1")

    def __repr__(self) -> str:
        args = f"length={self.length}"
        if self._generator_polynomial is not None:
            args += f", generator_polynomial={self.generator_polynomial}"
        if self._check_polynomial is not None:
            args += f", check_polynomial={self.check_polynomial}"
        args += f", systematic={self.systematic}"
        return f"{self.__class__.__name__}({args})"

    @property
    def length(self) -> int:
        return self._length

    @cached_property
    def generator_polynomial(self) -> BinaryPolynomial:
        if self._generator_polynomial is None:
            return self.modulus // self.check_polynomial
        return BinaryPolynomial(self._generator_polynomial)

    @cached_property
    def check_polynomial(self) -> BinaryPolynomial:
        if self._check_polynomial is None:
            return self.modulus // self.generator_polynomial
        return BinaryPolynomial(self._check_polynomial)

    @property
    def systematic(self) -> bool:
        return self._systematic

    @cached_property
    def modulus(self) -> BinaryPolynomial:
        return BinaryPolynomial.from_exponents([0, self.length])

    @property
    def dimension(self) -> int:
        return self.check_polynomial.degree

    @property
    def redundancy(self) -> int:
        return self.generator_polynomial.degree

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        # See [McE04, Sec. 8.1].
        n, k, m = self.length, self.dimension, self.redundancy
        generator_matrix = np.empty((k, n), dtype=int)
        X = BinaryPolynomial(0b10)  # The polynomial X
        if not self.systematic:
            for i in range(k):
                row_poly = X**i * self.generator_polynomial
                generator_matrix[i] = row_poly.coefficients(width=n)
        else:
            for i in range(k):
                row_poly = X ** (m + i) + X ** (m + i) % self.generator_polynomial
                generator_matrix[i] = row_poly.coefficients(width=n)
        return generator_matrix

    @cached_property
    def check_matrix(self) -> npt.NDArray[np.integer]:
        # See [McE04, Sec. 8.1].
        n, m = self.length, self.redundancy
        check_matrix = np.empty((m, n), dtype=int)
        X = BinaryPolynomial(0b10)  # The polynomial X
        if not self.systematic:
            for i in range(m):
                row_poly = X**i * self.check_polynomial.reciprocal()
                check_matrix[i] = row_poly.coefficients(width=n)
        else:
            for j in range(n):
                col_poly = X**j % self.generator_polynomial
                check_matrix[:, j] = col_poly.coefficients(width=m)
        return check_matrix

    @vectorized_method
    def _encode(self, u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        u_poly = BinaryPolynomial.from_coefficients(u)
        if not self.systematic:
            v_poly = u_poly * self.generator_polynomial
        else:
            u_poly_shifted = u_poly << self.redundancy
            b_poly = u_poly_shifted % self.generator_polynomial
            v_poly = u_poly_shifted + b_poly
        v = v_poly.coefficients(width=self.length)
        return v

    @vectorized_method
    def _inverse_encode(self, v: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        s = self._check(v)
        if not np.all(s == 0):
            raise ValueError("one or more inputs in 'v' are not valid codewords")
        v_poly = BinaryPolynomial.from_coefficients(v)
        if not self.systematic:
            u_poly = v_poly // self.generator_polynomial
        else:
            u_poly = v_poly >> self.redundancy
        u = u_poly.coefficients(width=self.dimension)
        return u

    @vectorized_method
    def _check(self, r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        r_poly = BinaryPolynomial.from_coefficients(r)
        s_poly = r_poly % self.generator_polynomial
        s = s_poly.coefficients(width=self.redundancy)
        return s
