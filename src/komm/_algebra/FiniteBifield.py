from functools import cached_property, reduce
from typing import Generic, Self, SupportsInt, TypeVar

import numpy as np
import numpy.typing as npt

from . import field
from .BinaryPolynomial import BinaryPolynomial, default_primitive_polynomial
from .Integers import mersenne_prime_factors

F = TypeVar("F", bound="FiniteBifield")


class FiniteBifieldElement(Generic[F]):
    def __init__(self, ambient: F, value: SupportsInt) -> None:
        self.ambient = ambient
        self.value = BinaryPolynomial(value)

    def __repr__(self) -> str:
        return bin(int(self))

    def __str__(self) -> str:
        return bin(int(self))

    def __int__(self) -> int:
        return int(self.value)

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
        return BinaryPolynomial.from_coefficients([int(c) for c in coefficients])


class FiniteBifield:
    r"""
    Finite field with binary characteristic. Objects of this class represent a *finite field* $\mathrm{GF}(2^k)$ (also known as *Galois field*), with *characteristic* $2$ and *degree* $k$.

    Parameters:
        degree: Degree $k$ of the finite field. Must be a positive integer.

        modulus: Modulus $p(X)$ of the field, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former. Must be an irreducible polynomial. If not specified, the modulus is chosen from [the list of default primitive polynomials](/res/primitive-polynomials).


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

    def __hash__(self) -> int:
        return hash((self.degree, self.modulus))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.degree == other.degree and self.modulus == other.modulus

    def __repr__(self) -> str:
        if self.modulus.value == default_primitive_polynomial(self.degree):
            args = f"{self.degree}"
        else:
            args = f"{self.degree}, modulus={self.modulus}"
        return f"{self.__class__.__name__}({args})"

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

    @cached_property
    def primitive_element(self) -> FiniteBifieldElement[Self]:
        r"""
        A primitive element of the finite field. It is a generator of the field's multiplicative group.

        Of all primitive elements, this property returns the one with the smallest integer representation, which is $X$ if the modulus is primitive.

        Examples:
            >>> field = komm.FiniteBifield(4)
            >>> field.primitive_element
            0b10

            >>> field = komm.FiniteBifield(4, modulus=0b11111)
            >>> field.primitive_element
            0b11
        """
        n = self.order - 1
        factors = set(mersenne_prime_factors(self.degree))
        return next(
            x
            for x in map(self, range(1, self.order))
            if all(x ** (n // q) != self.one for q in factors)
        )

    @cached_property
    def _exp_table(self) -> npt.NDArray[np.integer]:
        # α^i for i in [0 : 2n), avoiding mod n of indices.
        n = self.order - 1
        modulus, alpha = int(self.modulus), int(self.primitive_element)
        table = np.empty(2 * n, dtype=int)
        x = 1
        for i in range(n):
            table[i] = x
            x = multiply_mod(x, alpha, modulus)
        table[n:] = table[:n]
        return table

    @cached_property
    def _log_table(self) -> npt.NDArray[np.integer]:
        # log_α(x) for x in [1 : n], dummy at 0.
        n = self.order - 1
        table = np.zeros(self.order, dtype=int)
        table[self._exp_table[:n]] = np.arange(n)
        return table

    def multiply(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Multiplies elements of the finite field. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

        Parameters:
            x: The first factor.
            y: The second factor.

        Returns:
            product: The product $x y$.

        Examples:
            >>> field = komm.FiniteBifield(4)
            >>> field.multiply([0b1011, 0b0110], 0b1100)
            array([13, 14])
        """
        x, y = np.asarray(x), np.asarray(y)
        exp, log = self._exp_table, self._log_table
        return np.where((x == 0) | (y == 0), 0, exp[log[x] + log[y]])

    def divide(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Divides elements of the finite field. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

        Parameters:
            x: The dividend.
            y: The divisor.

        Returns:
            quotient: The quotient $x / y$.

        Raises:
            ZeroDivisionError: If the divisor is zero.

        Examples:
            >>> field = komm.FiniteBifield(4)
            >>> field.divide([0b1011, 0b0110], 0b1100)
            array([2, 9])
        """
        x, y = np.asarray(x), np.asarray(y)
        if np.any(y == 0):
            raise ZeroDivisionError("division by zero")
        n = self.order - 1
        exp, log = self._exp_table, self._log_table
        return np.where(x == 0, 0, exp[log[x] - log[y] + n])

    def power(
        self, x: npt.ArrayLike, exponent: npt.ArrayLike
    ) -> npt.NDArray[np.integer]:
        r"""
        Raises elements of the finite field to integer powers. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

        Parameters:
            x: The base.
            exponent: The exponent.

        Returns:
            power: The power $x^{\mathtt{exponent}}$.

        Raises:
            ZeroDivisionError: If the base is zero and the exponent is negative.

        Examples:
            >>> field = komm.FiniteBifield(4)
            >>> field.power(0b10, [0, 1, 2, 3, 4, 15, -1])
            array([1, 2, 4, 8, 3, 1, 9])
        """
        x, exponent = np.asarray(x), np.asarray(exponent)
        if np.any((x == 0) & (exponent < 0)):
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        n = self.order - 1
        exp, log = self._exp_table, self._log_table
        result = exp[log[x] * (exponent % n) % n]
        return np.where(x == 0, np.where(exponent == 0, 1, 0), result)


def multiply_mod(x: int, y: int, modulus: int) -> int:
    # Field product on integer representations.
    order = 1 << (modulus.bit_length() - 1)
    result = 0
    while y:
        if y & 1:
            result ^= x
        y >>= 1
        x <<= 1
        if x >= order:
            x ^= modulus
    return result


def horner(
    field: FiniteBifield,
    coefficients: npt.ArrayLike,
    points: npt.ArrayLike,
) -> npt.NDArray[np.integer]:
    r"""
    Evaluates polynomials with coefficients in a finite field, using Horner's method. Coefficients and points are given by their integer representations, in $[0 : 2^k)$.

    Parameters:
        field: Finite field.
        coefficients: Coefficients of the polynomials, in increasing order of degree along the last dimension.
        points: Points at which to evaluate the polynomials. Must be a 1D-array.

    Returns:
        values: Values of the polynomials at the points. Has the same shape as `coefficients`, but with the last dimension replaced by the number of points.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> coefficients = [1, 1, 0, 0b0110]  # 1 + X + α^5 X^3
        >>> points = [0, 1, 0b0111, 0b1000, 0b1111]  # 0, 1, α^10, α^3, α^12
        >>> horner(field, coefficients, points)
        array([1, 6, 0, 0, 0])
    """
    coefficients, points = np.asarray(coefficients), np.asarray(points)
    values = np.zeros(coefficients.shape[:-1] + points.shape, dtype=int)
    for i in reversed(range(coefficients.shape[-1])):
        values = field.multiply(values, points) ^ coefficients[..., i, np.newaxis]
    return values
