import functools
import operator
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from attrs import field, frozen
from typing_extensions import Self

from .._util.bit_operations import bits_to_int, int_to_bits
from . import domain, ring

T = TypeVar("T", bound=ring.RingElement)


@frozen
class BinaryPolynomial:
    r"""
    Binary polynomial. A *binary polynomial* is a polynomial whose coefficients are elements in the finite field $\mathbb{F}_2 = \\{ 0, 1 \\}$.

    The default constructor of the class expects the following:

    Attributes:
        value (int): An integer whose binary digits represent the coefficients of the polynomial—the leftmost bit standing for the highest degree term. For example, the binary polynomial $X^4 + X^3 + X$ is represented by the integer `0b11010` = `0o32` = `26`.

    Examples:
        >>> komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
        BinaryPolynomial(0b11010)

    See also the class methods [`from_coefficients`](./#from_coefficients) and [`from_exponents`](./#from_exponents) for alternative ways to construct a binary polynomial.

    <h2>Algebraic structure</h2>

    The binary polynomials form an *Euclidean domain*. The following operations are supported: addition (`+`), subtraction (`-`), multiplication (`*`), euclidean division (`//`), modulo (`%`), and exponentiation (`**`).

    Examples:
        >>> poly1 = komm.BinaryPolynomial(0b10111)  # X^4 + X^2 + X + 1
        >>> poly2 = komm.BinaryPolynomial(0b101)  # X^2 + 1
        >>> poly1 + poly2  # X^4 + X
        BinaryPolynomial(0b10010)
        >>> poly1 - poly2  # X^4 + X
        BinaryPolynomial(0b10010)
        >>> poly1 * poly2  # X^6 + X^3 + X + 1
        BinaryPolynomial(0b1001011)
        >>> poly1 // poly2  # X^2
        BinaryPolynomial(0b100)
        >>> poly1 % poly2  # X + 1
        BinaryPolynomial(0b11)
        >>> poly1 ** 2  # X^8 + X^4 + X^2 + 1
        BinaryPolynomial(0b100010101)
    """

    value: int = field(converter=int)

    @property
    def ambient(self):
        return BinaryPolynomials()

    @classmethod
    def from_coefficients(cls, coefficients: npt.ArrayLike) -> Self:
        r"""
        Constructs a binary polynomial from its coefficients.

        Parameters:
            coefficients (Array1D[int]): The coefficients of the binary polynomial—the $i$-th element of the array standing for the coefficient of $X^i$. For example, `[0, 1, 0, 1, 1]` represents the binary polynomial $X^4 + X^3 + X$.

        Examples:
            >>> komm.BinaryPolynomial.from_coefficients([0, 1, 0, 1, 1])  # X^4 + X^3 + X
            BinaryPolynomial(0b11010)
        """
        return cls(bits_to_int(coefficients))

    @classmethod
    def from_exponents(cls, exponents: npt.ArrayLike) -> Self:
        r"""
        Constructs a binary polynomial from its exponents.

        Parameters:
            exponents (Array1D[int]): The exponents of the nonzero terms of the binary polynomial. For example, `[1, 3, 4]` represents the binary polynomial $X^4 + X^3 + X$.

        Examples:
            >>> komm.BinaryPolynomial.from_exponents([1, 3, 4])  # X^4 + X^3 + X
            BinaryPolynomial(0b11010)
        """
        return cls(bits_to_int(np.bincount(exponents)))

    def __int__(self) -> int:
        return self.value

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, self.__class__):
            return self.value == other.value
        return NotImplemented

    def __lshift__(self, n: int) -> Self:
        return self.__class__(self.value << n)

    def __rshift__(self, n: int) -> Self:
        return self.__class__(self.value >> n)

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.value ^ other.value)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.value ^ other.value)

    def __neg__(self) -> Self:
        return self

    def __mul__(self, other: Self) -> Self:
        coeffs = np.convolve(self.coefficients(), other.coefficients()) % 2
        return self.__class__(bits_to_int(coeffs))

    def __rmul__(self, other: int) -> Self:
        if other % 2 == 0:
            return self.__class__(0)
        else:
            return self

    def __pow__(self, exponent: int) -> Self:
        return ring.power(self, exponent)

    def __divmod__(self, den: Self) -> tuple[Self, Self]:
        if den.value == 0:
            raise ZeroDivisionError("division by zero polynomial")
        div, mod = 0, self.value
        d = mod.bit_length() - den.value.bit_length()
        while d >= 0:
            div ^= 1 << d
            mod ^= den.value << d
            d = mod.bit_length() - den.value.bit_length()
        return self.__class__(div), self.__class__(mod)

    def __floordiv__(self, other: Self) -> Self:
        return self.__divmod__(other)[0]

    def __mod__(self, other: Self) -> Self:
        return self.__divmod__(other)[1]

    @property
    def degree(self) -> int:
        r"""
        The degree of the polynomial.

        Examples:
            >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
            >>> poly.degree
            4
        """
        return self.value.bit_length() - 1

    def coefficients(self, width: int | None = None) -> npt.NDArray[np.integer]:
        r"""
        Returns the coefficients of the binary polynomial.

        Parameters:
            width (int | None): If this parameter is specified, the output will be filled with zeros on the right so that the its length will be the specified value.

        Returns:
            coefficients (Array1D[int]): Coefficients of the binary polynomial. The $i$-th element of the array stands for the coefficient of $X^i$.

        Examples:
            >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
            >>> poly.coefficients()
            array([0, 1, 0, 1, 1])
            >>> poly.coefficients(width=8)
            array([0, 1, 0, 1, 1, 0, 0, 0])
        """
        return int_to_bits(self.value, width=width or max(self.degree + 1, 1))

    def exponents(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the exponents of the binary polynomial.

        Returns:
            exponents (Array1D[int]): Exponents of the nonzero terms of the binary polynomial. The exponents are returned in ascending order.

        Examples:
            >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
            >>> poly.exponents()
            array([1, 3, 4])
        """
        return np.flatnonzero(self.coefficients())

    def evaluate(self, point: T) -> T:
        r"""
        Evaluates the polynomial at a given point. Uses Horner's method.

        Parameters:
            point (RingElement): The point at which the polynomial is evaluated. It must be an element of a ring in which multiplication by integers is defined.

        Returns:
            result (RingElement): The result of evaluating the binary polynomial at `point`. It has the same type as `point`.

        Examples:
            >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
            >>> poly.evaluate(komm.Integer(7))  # same as 7**4 + 7**3 + 7
            Integer(value=2751)
        """
        return ring.binary_horner(self.coefficients(), point)

    def __repr__(self) -> str:
        return f"BinaryPolynomial({self.value:#b})"

    def __str__(self) -> str:
        return bin(self.value)

    @classmethod
    def xgcd(cls, poly1: Self, poly2: Self) -> tuple[Self, Self, Self]:
        r"""
        Performs the extended Euclidean algorithm on two given binary polynomials.
        """
        return domain.xgcd(poly1, poly2)

    @classmethod
    def gcd(cls, *poly_list: Self) -> Self:
        r"""
        Computes the greatest common divisor (gcd) of the arguments.
        """
        return functools.reduce(domain.gcd, poly_list)

    @classmethod
    def lcm(cls, *poly_list: Self) -> Self:
        r"""
        Computes the least common multiple (lcm) of the arguments.
        """
        return functools.reduce(operator.mul, poly_list) // cls.gcd(*poly_list)


@frozen
class BinaryPolynomials:
    def __call__(self, value: int) -> BinaryPolynomial:
        return BinaryPolynomial(value)

    @property
    def zero(self) -> BinaryPolynomial:
        return BinaryPolynomial(0)

    @property
    def one(self) -> BinaryPolynomial:
        return BinaryPolynomial(1)


def default_primitive_polynomial(degree: int) -> BinaryPolynomial:
    if not 1 <= degree <= 16:
        raise ValueError("only degrees in the range [1 : 16] are implemented")
    return BinaryPolynomial(
        {
            1: 0b11,
            2: 0b111,
            3: 0b1011,
            4: 0b10011,
            5: 0b100101,
            6: 0b1000011,
            7: 0b10001001,
            8: 0b100011101,
            9: 0b1000010001,
            10: 0b10000001001,
            11: 0b100000000101,
            12: 0b1000001010011,
            13: 0b10000000011011,
            14: 0b100010001000011,
            15: 0b1000000000000011,
            16: 0b10000000010000011,
        }[degree]
    )
