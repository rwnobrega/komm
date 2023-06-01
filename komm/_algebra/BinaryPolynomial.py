import functools
import operator
import numpy as np

from .._util import \
    _int2binlist, _binlist2int

from .util import \
    gcd, power, xgcd, binary_horner

class BinaryPolynomial:
    """
    Binary polynomial. A *binary polynomial* is a polynomial whose coefficients are elements in the finite field :math:`\\mathbb{F}_2 = \\{ 0, 1 \\}`. The default constructor takes an :obj:`int` as input, whose binary digits represent the coefficients of the polynomial---the leftmost bit standing for the highest degree term. For example, the binary polynomial :math:`X^4 + X^3 + X` is represented by the integer :code:`0b11010` = :code:`0o32` = :code:`26`. There are two alternative constructors for this class, the class methods :func:`from_coefficients` and :func:`from_exponents`.  See their documentation for details.

    This class supports addition, multiplication, and division of binary polynomials.

    .. rubric:: Examples

    >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
    >>> poly
    BinaryPolynomial(0b11010)

    >>> poly1 = komm.BinaryPolynomial(0b10100)  # X^4 + X^2
    >>> poly2 = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
    >>> poly1 + poly2  # X^3 + X^2 + X
    BinaryPolynomial(0b1110)
    >>> poly1 * poly2  # X^8 + X^7 + X^6 + X^3
    BinaryPolynomial(0b111001000)
    >>> poly1**2  # X^8 + X^4
    BinaryPolynomial(0b100010000)
    """
    def __init__(self, integer):
        self._integer = int(integer)

    @classmethod
    def from_coefficients(cls, coefficients):
        """
        Constructs a :obj:`BinaryPolynomial` from its coefficients. It expects the following parameter:

        :code:`coefficients` : 1D-array of :obj:`int`
            The coefficients of the binary polynomial---the :math:`i`-th element of the array standing for the coefficient of :math:`X^i`. For example, :code:`[0, 1, 0, 1, 1]` represents the binary polynomial :math:`X^4 + X^3 + X`.

        .. rubric:: Examples

        >>> komm.BinaryPolynomial.from_coefficients([0, 1, 0, 1, 1])  # X^4 + X^3 + X
        BinaryPolynomial(0b11010)
        """
        return cls(_binlist2int(coefficients))

    @classmethod
    def from_exponents(cls, exponents):
        """
        Constructs a :obj:`BinaryPolynomial` from its exponents. It expects the following parameter:

        :code:`coefficients` : 1D-array of :obj:`int`
            The exponents of the nonzero terms of the binary polynomial. For example, :code:`[1, 3, 4]` represents the binary polynomial :math:`X^4 + X^3 + X`.

        .. rubric:: Examples

        >>> komm.BinaryPolynomial.from_exponents([1, 3, 4])  # X^4 + X^3 + X
        BinaryPolynomial(0b11010)
        """
        return cls(_binlist2int(np.bincount(exponents)))

    @property
    def degree(self):
        """
        The degree of the polynomial. This property is read-only.

        .. rubric:: Examples

        >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
        >>> poly.degree
        4
        """
        return self._integer.bit_length() - 1

    def coefficients(self, width=None):
        """
        Returns the coefficients of the binary polynomial.

        .. rubric:: Input

        :code:`width` : :obj:`int`, optional
            If this parameter is specified, the output will be filled with zeros on the right so that the its length will be the specified value.

        .. rubric:: Output

        :code:`coefficients` : 1D-array of :obj:`int`
            Coefficients of the binary polynomial. The :math:`i`-th element of the array stands for the coefficient of :math:`X^i`.

        .. rubric:: Examples

        >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
        >>> poly.coefficients()
        array([0, 1, 0, 1, 1])
        >>> poly.coefficients(width=8)
        array([0, 1, 0, 1, 1, 0, 0, 0])
        """
        return np.array(_int2binlist(self._integer, width=width), dtype=int)

    def exponents(self):
        """
        Returns the exponents of the binary polynomial.

        .. rubric:: Output

        :code:`exponents` : 1D-array of :obj:`int`
            Exponents of the nonzero terms of the binary polynomial. The exponents are returned in ascending order.

        .. rubric:: Examples

        >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
        >>> poly.exponents()
        array([1, 3, 4])
        """
        return np.flatnonzero(self.coefficients())

    def __int__(self):
        return self._integer

    def __hash__(self):
        return self._integer

    def __eq__(self, other):
        return int(self) == int(other)

    def __lshift__(self, n):
        return self.__class__(self._integer.__lshift__(n))

    def __rshift__(self, n):
        return self.__class__(self._integer.__rshift__(n))

    def __add__(self, other):
        return self.__class__(self._integer.__xor__(other._integer))

    def __sub__(self, other):
        return self.__class__(self._integer.__xor__(other._integer))

    def __mul__(self, other):
        return self.from_coefficients(np.convolve(self.coefficients(), other.coefficients()) % 2)

    def __pow__(self, exponent):
        return power(self, exponent, self.__class__)

    def __divmod__(self, den):
        div, mod, den = 0, self._integer, den._integer
        d = mod.bit_length() - den.bit_length()
        while d >= 0:
            div ^= (1 << d)
            mod ^= (den << d)
            d = mod.bit_length() - den.bit_length()
        return self.__class__(div), self.__class__(mod)

    def __floordiv__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        return self.__divmod__(other)[1]

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

        >>> poly = komm.BinaryPolynomial(0b11010)  # X^4 + X^3 + X
        >>> poly.evaluate(7)  # same as 7**4 + 7**3 + 7
        2751
        >>> point = np.array([[1, 2], [3, 4]])
        >>> poly.evaluate(point)  # same as point**4 + point**3 + point
        array([[  3,  26],
               [111, 324]])
        """
        return binary_horner(self, point)

    def __repr__(self):
        args = '{}'.format(bin(self._integer))
        return '{}({})'.format(self.__class__.__name__, args)

    def __str__(self):
        return bin(self._integer)

    @classmethod
    def xgcd(cls, poly1, poly2):
        """
        Performs the extended Euclidean algorithm on two given binary polynomials.
        """
        return xgcd(poly1, poly2, cls)

    @classmethod
    def gcd(cls, *poly_list):
        """
        Computes the greatest common divisor (gcd) of the arguments.
        """
        return functools.reduce(functools.partial(gcd, ring=cls), poly_list)

    @classmethod
    def lcm(cls, *poly_list):
        """
        Computes the least common multiple (lcm) of the arguments.
        """
        return functools.reduce(operator.mul, poly_list) // cls.gcd(*poly_list)
