import functools
import operator

from fractions import Fraction

import numpy as np

from .util import \
    gcd, power, horner

class RationalPolynomial:
    """
    Rational polynomial. A *rational polynomial* is a polynomial whose coefficients are all rational numbers.

    .. rubric:: Examples

    >>> from fractions import Fraction
    >>> poly = komm.RationalPolynomial(['1/2', '0', '3'])  # 1/2 + 3 X^2
    >>> poly
    RationalPolynomial(['1/2', '0', '3'])
    """
    def __init__(self, coefficients):
        if isinstance(coefficients, (int, Fraction)):
            coefficients = [Fraction(coefficients)]
        elif isinstance(coefficients, self.__class__):
            coefficients = coefficients._coefficients

        self._coefficients = np.array(np.trim_zeros([Fraction(x) for x in coefficients], trim='b'))

    @classmethod
    def monomial(cls, degree, coefficient=1):
        """
        Constructs a monomial. This is an polynomial of the form :math:`cX^d`. It expects the following parameters:

        :code:`degree` : :obj:`int`
            The degree :math:`d` of the monomial.

        :code:`coefficient` : :obj:`int`, optional
            The coefficient :math:`c` of the monomial. The default value is :math:`1`.

        .. rubric:: Examples

        >>> komm.RationalPolynomial.monomial(4, 2)  # 2 X^4
        RationalPolynomial(['0', '0', '0', '0', '2'])
        """
        return cls([0] * degree + [coefficient])

    def coefficients(self, width=None):
        """
        Returns the coefficients of the polynomial.

        .. rubric:: Input

        :code:`width` : :obj:`int`, optional
            If this parameter is specified, the output will be filled with zeros on the right so that the its length will be the specified value.

        .. rubric:: Output

        :code:`coefficients` : 1D-array of :obj:`int`
            Coefficients of the polynomial. The :math:`i`-th element of the array stands for the coefficient of :math:`X^i`.

        .. rubric:: Examples

        >>> poly = komm.RationalPolynomial(['0', '1/3', '2/3'])  # (1/3) X + (2/3) X^2
        >>> poly.coefficients()
        array([Fraction(0, 1), Fraction(1, 3), Fraction(2, 3)], dtype=object)
        >>> poly.coefficients(width=5)
        array([Fraction(0, 1), Fraction(1, 3), Fraction(2, 3), Fraction(0, 1),
               Fraction(0, 1)], dtype=object)
        """
        if width is None:
            coefficients = self._coefficients
        else:
            coefficients = np.empty((width, ), dtype=object)
            coefficients[:self._coefficients.size] = self._coefficients
            coefficients[self._coefficients.size:] = Fraction(0)
        return coefficients

    @property
    def degree(self):
        """
        The degree of the polynomial. This property is read-only.

        .. rubric:: Examples

        >>> poly = komm.RationalPolynomial([1, 0, 3])  # 1 + 3X^2
        >>> poly.degree
        2
        """
        return self._coefficients.size - 1

    def __eq__(self, other):
        return np.array_equal(self._coefficients, other._coefficients)

    def __add__(self, other):
        if self.degree > other.degree:
            return self.__class__(self._coefficients + np.pad(other._coefficients, (0, self.degree - other.degree), mode='constant'))
        else:
            return self.__class__(np.pad(self._coefficients, (0, other.degree - self.degree), mode='constant') + other._coefficients)

    def __sub__(self, other):
        if self.degree > other.degree:
            return self.__class__(self._coefficients - np.pad(other._coefficients, (0, self.degree - other.degree), mode='constant'))
        else:
            return self.__class__(np.pad(self._coefficients, (0, other.degree - self.degree), mode='constant') - other._coefficients)

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
        """
        Evaluates the polynomial at a given point. Uses Horner's method.

        .. rubric:: Input

        :code:`point` : ring-like type
            Any Python object supporting the operations of addition, subtraction, and multiplication.

        .. rubric:: Output

        :code:`result` : ring-like type
            The result of evaluating the binary polynomial at :code:`point`. It has the same type as :code:`point`.

        .. rubric:: Examples

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
        args = '{}'.format([str(f) for f in self._coefficients])
        return '{}({})'.format(self.__class__.__name__, args)

    @classmethod
    def gcd(cls, *poly_list):
        """
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
        """
        Computes the least common multiple (lcm) of the arguments.
        """
        return functools.reduce(operator.mul, poly_list) // cls.gcd(*poly_list)
