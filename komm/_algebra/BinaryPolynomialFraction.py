from . import BinaryPolynomial

from .util import power

class BinaryPolynomialFraction:
    """
    Binary polynomial fraction. A *binary polynomial fraction* is a ratio of two binary polynomials (:class:`BinaryPolynomial`).
    """
    def __init__(self, numerator, denominator=0b1):
        self._numerator = BinaryPolynomial(numerator)
        self._denominator = BinaryPolynomial(denominator)
        if denominator == 0:
            raise ZeroDivisionError('Denominator cannot be zero')
        self._reduce_to_lowest_terms()

    def _reduce_to_lowest_terms(self):
        gcd = BinaryPolynomial.gcd(self._numerator, self._denominator)
        self._numerator //= gcd
        self._denominator //= gcd

    def __repr__(self):
        args = '{}, {}'.format(self._numerator, self._denominator)
        return '{}({})'.format(self.__class__.__name__, args)

    def __str__(self):
        if self._denominator == 0b1:
            return str(self._numerator)
        else:
            return str(self._numerator) + '/' + str(self._denominator)

    @property
    def numerator(self):
        """
        The numerator of the polynomial fraction.
        """
        return self._numerator

    @property
    def denominator(self):
        """
        The denominator of the polynomial fraction.
        """
        return self._denominator

    def __add__(self, other):
        numerator = self._numerator * other._denominator + self._denominator * other._numerator
        denominator = self._denominator * other._denominator
        return self.__class__(numerator, denominator)

    def __sub__(self, other):
        numerator = self._numerator * other._denominator - self._denominator * other._numerator
        denominator = self._denominator * other._denominator
        return self.__class__(numerator, denominator)

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
        return self._numerator * other._denominator == self._denominator * other._numerator

    def inverse(self):
        """
        Returns the multiplicative inverse the polynomial fraction.
        """
        return self.__class__(self._denominator, self._numerator)
