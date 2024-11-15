from fractions import Fraction

import numpy as np

from .field import power
from .RationalPolynomial import RationalPolynomial


class RationalPolynomialFraction:
    r"""
    Rational polynomial fraction. A *rational polynomial fraction* is a ratio of two [rational polynomials](/ref/RationalPolynomial).
    """

    def __init__(
        self,
        numerator: int | RationalPolynomial,
        denominator: int | RationalPolynomial = 1,
    ):
        self._numerator = RationalPolynomial(numerator)
        self._denominator = RationalPolynomial(denominator)
        if self._denominator.degree == -1:
            raise ZeroDivisionError("denominator cannot be zero")
        self._reduce_to_lowest_terms()
        self._reduce_to_integer_coefficients()

    @classmethod
    def monomial(cls, degree, coefficient=1):
        return cls(RationalPolynomial.monomial(degree, coefficient))

    def _reduce_to_lowest_terms(self):
        gcd = RationalPolynomial.gcd(self._numerator, self._denominator)
        self._numerator //= gcd
        self._denominator //= gcd

    def _reduce_to_integer_coefficients(self):
        all_denominators = [x.denominator for x in self._numerator._coefficients] + [
            x.denominator for x in self._denominator._coefficients
        ]
        a = np.lcm.reduce([n for n in all_denominators if n != 0])
        self._numerator *= RationalPolynomial([Fraction(a, 1)])
        self._denominator *= RationalPolynomial([Fraction(a, 1)])

        all_numerators = [x.numerator for x in self._numerator._coefficients] + [
            x.numerator for x in self._denominator._coefficients
        ]
        b = np.gcd.reduce([n for n in all_numerators if n != 0])
        self._numerator *= RationalPolynomial([Fraction(1, b)])
        self._denominator *= RationalPolynomial([Fraction(1, b)])

    def __repr__(self):
        args = "{}, {}".format(self._numerator, self._denominator)
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def __add__(self, other):
        numerator = (
            self._numerator * other._denominator + self._denominator * other._numerator
        )
        denominator = self._denominator * other._denominator
        return self.__class__(numerator, denominator)

    def __sub__(self, other):
        numerator = (
            self._numerator * other._denominator - self._denominator * other._numerator
        )
        denominator = self._denominator * other._denominator
        return self.__class__(numerator, denominator)

    def __neg__(self):
        return self.__class__(-self._numerator, self._denominator)

    def __mul__(self, other):
        numerator = self._numerator * other._numerator
        denominator = self._denominator * other._denominator
        return self.__class__(numerator, denominator)

    def __truediv__(self, other):
        numerator = self._numerator * other._denominator
        denominator = self._denominator * other._numerator
        return self.__class__(numerator, denominator)

    def __pow__(self, exponent):
        return power(self, exponent, self.__class__)

    def __eq__(self, other):
        return (
            self._numerator * other._denominator == self._denominator * other._numerator
        )

    def inverse(self):
        return self.__class__(self._denominator, self._numerator)
