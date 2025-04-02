from collections.abc import Sequence
from functools import reduce
from typing import Generic, SupportsInt, TypeVar

import numpy as np
from typing_extensions import Self

from . import field
from .BinaryPolynomial import BinaryPolynomial, default_primitive_polynomial

F = TypeVar("F", bound="FiniteBifield")


class FiniteBifieldElement(Generic[F]):
    def __init__(self, ambient: F, value: SupportsInt) -> None:
        self.ambient = ambient
        self.value = BinaryPolynomial(value)

    def __repr__(self) -> str:
        return bin(self.value.value)

    def __str__(self) -> str:
        return bin(self.value.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.ambient == other.ambient and self.value == other.value

    def _check_same_ambient(self, other: Self):
        if self.ambient != other.ambient:
            raise ValueError("elements must belong to the same finite field")

    def __add__(self, other: Self) -> Self:
        self._check_same_ambient(other)
        return self.__class__(self.ambient, self.value + other.value)

    def __sub__(self, other: Self) -> Self:
        self._check_same_ambient(other)
        return self.__class__(self.ambient, self.value - other.value)

    def __neg__(self) -> Self:
        return self

    def __mul__(self, other: Self) -> Self:
        self._check_same_ambient(other)
        value = (self.value * other.value) % self.ambient.modulus
        return self.__class__(self.ambient, value)

    def __rmul__(self, other: int) -> Self:
        if other % 2 == 0:
            return self.__class__(self.ambient, 0)
        else:
            return self

    def inverse(self) -> Self:
        d, s, _ = BinaryPolynomial.xgcd(self.value, self.ambient.modulus)
        if d.value == 1:
            return self.__class__(self.ambient, s)
        raise ZeroDivisionError("element does not have a multiplicative inverse")

    def __truediv__(self, other: Self) -> Self:
        self._check_same_ambient(other)
        return self * other.inverse()

    def __pow__(self, exponent: int) -> Self:
        return field.power(self, exponent)

    def logarithm(self, base: Self) -> int:
        for i in range(self.ambient.order):
            if base**i == self:
                return i
        raise ValueError("element is not a power of the base")

    def conjugates(self) -> list[Self]:
        conjugate_list: list[Self] = []
        exponent = 0
        while True:
            y = self ** (2**exponent)
            if y not in conjugate_list:
                conjugate_list.append(y)
            else:
                break
            exponent += 1
        return conjugate_list

    def minimal_polynomial(self) -> BinaryPolynomial:
        one = self.ambient.one
        monomials = [np.array([y, one], dtype=object) for y in self.conjugates()]
        coefficients: list[Self] = list(reduce(np.convolve, monomials))
        return BinaryPolynomial.from_coefficients([c.value.value for c in coefficients])


class FiniteBifield:
    r"""
    Finite field with binary characteristic. Objects of this class represent a *finite field* $\mathrm{GF}(2^k)$ (also known as *Galois field*), with *characteristic* $2$ and *degree* $k$.

    Parameters:
        degree: Degree $k$ of the finite field. Must be a positive integer.

        modulus: Modulus $p(X)$ of the field, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former. Must be an irreducible polynomial. If not specified, the modulus is chosen from [the list of default primitive polynomials](/resources/primitive-polynomials).


    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> field
        FiniteBifield(4)
        >>> (field.characteristic, field.degree, field.order)
        (2, 4, 16)
        >>> field.modulus
        BinaryPolynomial(0b10011)

        >>> field = komm.FiniteBifield(4, modulus=0b11001)
        >>> field
        FiniteBifield(4, modulus=0b11001)
        >>> (field.characteristic, field.degree, field.order)
        (2, 4, 16)
        >>> field.modulus
        BinaryPolynomial(0b11001)

    <h2>Construction of elements</h2>

    To construct *elements* of the finite field, call the finite field object. For example, `field(0b1101)` will construct the element whose polynomial representation is $X^3 + X^2 + 1$.

    <h2>Algebraic structure</h2>

    The following operations are supported: addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), and exponentiation (`**`).

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> x = field(0b1011)
        >>> y = field(0b1100)
        >>> x + y
        0b111
        >>> x - y
        0b111
        >>> x * y
        0b1101
        >>> x / y
        0b10
        >>> x**2
        0b1001

    <h2>Further methods on elements</h2>

    The following methods are available on elements of the finite field:

    - `logarithm(base)`: Returns the logarithm of the element, with respect to a given base.
    - `conjugates()`: Returns the conjugates of the element.
    - `minimal_polynomial()`: Returns the minimal polynomial of the element.

    For more details, see <cite>LC04, Sec. 2.5</cite>.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> x = field(0b1011)
        >>> base = field(0b10)
        >>> x.logarithm(base)
        7
        >>> x.conjugates()
        [0b1011, 0b1001, 0b1101, 0b1110]
        >>> x.minimal_polynomial()
        BinaryPolynomial(0b11001)
    """

    def __init__(self, degree: int, modulus: BinaryPolynomial | int | None = None):
        if degree < 1:
            raise ValueError("'degree' must be a positive integer")
        self.degree = degree
        if modulus is None:
            self.modulus = default_primitive_polynomial(degree)
        else:
            self.modulus = BinaryPolynomial(modulus)
        if self.modulus.degree != self.degree:
            raise ValueError("'modulus' must have the same degree as the field")
        if not self.modulus.is_irreducible():
            raise ValueError("'modulus' must be an irreducible polynomial")

    def __call__(self, value: int | BinaryPolynomial) -> FiniteBifieldElement[Self]:
        return FiniteBifieldElement(self, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.degree == other.degree and self.modulus == other.modulus

    @property
    def zero(self) -> FiniteBifieldElement[Self]:
        return self(0)

    @property
    def one(self) -> FiniteBifieldElement[Self]:
        return self(1)

    @property
    def characteristic(self) -> int:
        r"""
        The characteristic $2$ of the finite field.
        """
        return 2

    @property
    def order(self) -> int:
        r"""
        The order (number of elements) of the finite field. It is given by $2^k$.
        """
        return 2**self.degree

    def __repr__(self) -> str:
        if self.modulus.value == default_primitive_polynomial(self.degree):
            args = f"{self.degree}"
        else:
            args = f"{self.degree}, modulus={self.modulus}"
        return f"{self.__class__.__name__}({args})"


def find_roots(
    field: F,
    coefficients: Sequence[FiniteBifieldElement[F]],
) -> list[FiniteBifieldElement[F]]:
    r"""
    Returns the roots of a polynomial with coefficients in a finite field. This function uses exhaustive search to find the roots.

    Parameters:
        field: Finite field.
        coefficients: Coefficients of the polynomial, in increasing order of degree.

    Returns:
        List of roots of the polynomial.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> alpha = field(0b10)  # alpha = X, a primitive element
        >>> coefficients = [field.one, field.one, field.zero, alpha**5]  # 1 + X + alpha^5 X^3
        >>> find_roots(field, coefficients)
        [0b111, 0b1000, 0b1111]
        >>> [alpha**10, alpha**3, alpha**12]
        [0b111, 0b1000, 0b1111]
    """
    roots: list[FiniteBifieldElement[F]] = []
    for i in range(field.order):
        x = field(i)
        evaluated = field.zero
        for coefficient in reversed(coefficients):  # Horner's method
            evaluated = evaluated * x + coefficient
        if evaluated == field.zero:
            roots.append(x)
            if len(roots) >= len(coefficients) - 1:
                break
    return roots
