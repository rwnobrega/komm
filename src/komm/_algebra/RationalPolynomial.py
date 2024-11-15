import functools
import operator
from fractions import Fraction

import numpy as np

from ._util import gcd, horner
from .ring import power


class RationalPolynomial:
    r"""
    Rational polynomial. A *rational polynomial* is a polynomial whose coefficients are all rational numbers. This class supports addition, subtraction, multiplication, division, and exponentiation.

    Examples:
        >>> poly1 = komm.RationalPolynomial(['1/2', '0', '3'])  # 1/2 + 3 X^2
        >>> poly1
        RationalPolynomial(['1/2', '0', '3'])
        >>> poly2 = komm.RationalPolynomial(['1/3', '2/3'])  # 1/3 + (2/3) X
        >>> poly2
        RationalPolynomial(['1/3', '2/3'])
        >>> poly1 + poly2  # 5/6 + (2/3) X + 3 X^2
        RationalPolynomial(['5/6', '2/3', '3'])
        >>> poly1 * poly2  # 1/6 + (1/3) X + X^2 + 2 X^3
        RationalPolynomial(['1/6', '1/3', '1', '2'])
    """

    def __init__(self, coefficients):
        r"""
        Constructor for the class.

        Parameters:
            coefficients (Array1D[int | str | Fraction]): The coefficients of the rational polynomial—the $i$-th element of the array standing for the coefficient of $X^i$. For example, `['1/2', '0', '3']` represents the rational polynomial $1/2 + 3 X^2$.

        Examples:
            >>> komm.RationalPolynomial(['1/2', '0', '3'])  # 1/2 + 3 X^2
            RationalPolynomial(['1/2', '0', '3'])
        """

        if isinstance(coefficients, (int, Fraction)):
            coefficients = [Fraction(coefficients)]
        elif isinstance(coefficients, self.__class__):
            coefficients = coefficients._coefficients

        self._coefficients = np.array(
            np.trim_zeros([Fraction(x) for x in coefficients], trim="b")
        )

    @classmethod
    def monomial(cls, degree, coefficient=1):
        r"""
        Constructs a monomial. This is an polynomial of the form $cX^d$.

        Parameters:
            degree (int): The degree $d$ of the monomial.

            coefficient (Optional[int]): The coefficient $c$ of the monomial. The default value is $1$.

        Examples:
            >>> komm.RationalPolynomial.monomial(4, 2)  # 2 X^4
            RationalPolynomial(['0', '0', '0', '0', '2'])
        """
        return cls([0] * degree + [coefficient])

    def coefficients(self, width=None):
        r"""
        Returns the coefficients of the polynomial.

        Parameters:
            width (Optional[int]): If this parameter is specified, the output will be filled with zeros on the right so that the its length will be the specified value.

        Returns:
            coefficients (Array1D[int]): Coefficients of the polynomial. The $i$-th element of the array stands for the coefficient of $X^i$.

        Examples:
            >>> poly = komm.RationalPolynomial(['0', '1/3', '2/3'])  # (1/3) X + (2/3) X^2
            >>> poly.coefficients()
            array([Fraction(0, 1), Fraction(1, 3), Fraction(2, 3)], dtype=object)
            >>> poly.coefficients(width=5)  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(0, 1), Fraction(1, 3), Fraction(2, 3), Fraction(0, 1), Fraction(0, 1)], dtype=object)
        """
        if width is None:
            coefficients = self._coefficients
        else:
            coefficients = np.empty((width,), dtype=object)
            coefficients[: self._coefficients.size] = self._coefficients
            coefficients[self._coefficients.size :] = Fraction(0)
        return coefficients

    @property
    def degree(self):
        r"""
        The degree of the polynomial.

        Examples:
            >>> poly = komm.RationalPolynomial([1, 0, 3])  # 1 + 3 X^2
            >>> poly.degree
            2
        """
        return self._coefficients.size - 1

    def __eq__(self, other):
        return np.array_equal(self._coefficients, other._coefficients)

    def __add__(self, other):
        if self.degree > other.degree:
            return self.__class__(
                self._coefficients
                + np.pad(
                    other._coefficients,
                    (0, self.degree - other.degree),
                    mode="constant",
                )
            )
        else:
            return self.__class__(
                np.pad(
                    self._coefficients, (0, other.degree - self.degree), mode="constant"
                )
                + other._coefficients
            )

    def __sub__(self, other):
        if self.degree > other.degree:
            return self.__class__(
                self._coefficients
                - np.pad(
                    other._coefficients,
                    (0, self.degree - other.degree),
                    mode="constant",
                )
            )
        else:
            return self.__class__(
                np.pad(
                    self._coefficients, (0, other.degree - self.degree), mode="constant"
                )
                - other._coefficients
            )

    def __neg__(self):
        return self.__class__(-self._coefficients)

    def __mul__(self, other):
        if self.degree == -1 or other.degree == -1:
            return self.__class__(0)
        return self.__class__(np.convolve(self._coefficients, other._coefficients))

    def __pow__(self, exponent):
        return power(self, exponent, self.__class__)

    def __divmod__(self, other):
        if other.degree == -1:
            raise ZeroDivisionError
        remainder = self._coefficients.tolist()
        quotient = [0] * (self.degree - other.degree + 1)
        for i in range(len(quotient)):
            quotient[-i - 1] = remainder[-1] / other._coefficients[-1]
            for j in range(1, len(other._coefficients)):
                remainder[-j - 1] -= quotient[-i - 1] * other._coefficients[-j - 1]
            del remainder[-1]
        return self.__class__(quotient), self.__class__(remainder)

    def __floordiv__(self, other):
        return divmod(self, other)[0]

    def __mod__(self, other):
        return divmod(self, other)[1]

    def evaluate(self, point):
        r"""
        Evaluates the polynomial at a given point. Uses Horner's method.

        Parameters:
            point (RingElement): Any Python object supporting the operations of addition, subtraction, and multiplication.

        Returns:
            result (RingElement): The result of evaluating the binary polynomial at `point`. It has the same type as `point`.

        Examples:
            >>> poly = komm.RationalPolynomial([0, 1, 0, -1, 2])  # X - X^3 + 2 X^4
            >>> poly.evaluate(7)  # same as 7 - 7**3 + 2 * 7**4
            Fraction(4466, 1)
            >>> point = np.array([[1, 2], [3, 4]])
            >>> poly.evaluate(point)  # same as point - point**3 + 2 * point**4
            array([[Fraction(2, 1), Fraction(26, 1)],
                   [Fraction(138, 1), Fraction(452, 1)]], dtype=object)
        """
        return horner(self, point)

    def __repr__(self):
        args = "{}".format([str(f) for f in self._coefficients])
        return "{}({})".format(self.__class__.__name__, args)

    @classmethod
    def gcd(cls, *poly_list):
        r"""
        Computes the greatest common divisor (gcd) of the arguments.
        """
        ans = functools.reduce(functools.partial(gcd, ring=cls), poly_list)
        a = np.lcm.reduce([coeff.denominator for coeff in ans._coefficients])
        ans *= cls([Fraction(a, 1)])
        b = np.gcd.reduce([coeff.numerator for coeff in ans._coefficients])
        ans *= cls([Fraction(1, b)])
        return ans

    @classmethod
    def lcm(cls, *poly_list):
        r"""
        Computes the least common multiple (lcm) of the arguments.
        """
        return functools.reduce(operator.mul, poly_list) // cls.gcd(*poly_list)
