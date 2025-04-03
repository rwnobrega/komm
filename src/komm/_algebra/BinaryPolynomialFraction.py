from typing import SupportsInt

from typing_extensions import Self

from . import field
from .BinaryPolynomial import BinaryPolynomial


class BinaryPolynomialFraction:
    r"""
    Binary polynomial fraction. A *binary polynomial fraction* is a ratio of two [binary polynomials](/ref/BinaryPolynomial).

    Parameters:
        numerator: The numerator of the fraction.
        denominator: The denominator of the fraction.

    Examples:
        >>> komm.BinaryPolynomialFraction(0b11010, 0b101)  # (X^4 + X^3 + X) / (X^2 + 1)
        BinaryPolynomialFraction(0b11010, 0b101)

    <h2>Algebraic structure</h2>

    The binary polynomial fractions form a *field*. The following operations are supported: addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), and exponentiation (`**`).
    """

    numerator: BinaryPolynomial
    denominator: BinaryPolynomial

    def __init__(self, numerator: SupportsInt, denominator: SupportsInt = 0b1) -> None:
        self.numerator = BinaryPolynomial(numerator)
        self.denominator = BinaryPolynomial(denominator)
        self.__post_init__()

    def __post_init__(self):
        if self.denominator == 0b0:
            raise ZeroDivisionError("denominator cannot be zero")
        gcd = BinaryPolynomial.gcd(self.numerator, self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.numerator}, {self.denominator})"

    def __str__(self) -> str:
        return str(self.numerator) + "/" + str(self.denominator)

    @property
    def ambient(self):
        return BinaryPolynomialFractions()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.numerator == other.numerator and self.denominator == other.denominator
        )

    def __add__(self, other: Self) -> Self:
        n = self.numerator * other.denominator + other.numerator * self.denominator
        d = self.denominator * other.denominator
        return self.__class__(n, d)

    def __sub__(self, other: Self) -> Self:
        n = self.numerator * other.denominator - other.numerator * self.denominator
        d = self.denominator * other.denominator
        return self.__class__(n, d)

    def __neg__(self) -> Self:
        return self

    def __mul__(self, other: Self) -> Self:
        n = self.numerator * other.numerator
        d = self.denominator * other.denominator
        return self.__class__(n, d)

    def __rmul__(self, other: int) -> Self:
        return self.__class__(other * self.numerator, self.denominator)

    def __truediv__(self, other: Self) -> Self:
        n = self.numerator * other.denominator
        d = self.denominator * other.numerator
        return self.__class__(n, d)

    def __pow__(self, exponent: int) -> Self:
        return field.power(self, exponent)

    def inverse(self) -> Self:
        return self.__class__(self.denominator, self.numerator)


class BinaryPolynomialFractions:
    def __call__(self, value: tuple[int, int]) -> BinaryPolynomialFraction:
        return BinaryPolynomialFraction(*value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return True

    @property
    def zero(self) -> BinaryPolynomialFraction:
        return BinaryPolynomialFraction(0, 1)

    @property
    def one(self) -> BinaryPolynomialFraction:
        return BinaryPolynomialFraction(1, 1)

    @property
    def characteristic(self) -> int:
        return 2
