import functools

import numpy as np

from . import BinaryPolynomial
from ._util import power


class FiniteBifield:
    r"""
    Finite field with binary characteristic. Objects of this class represent a *finite field* $\mathrm{GF}(2^k)$ (also known as *Galois field*), with *characteristic* $2$ and *degree* $k$.

    To construct *elements* of the finite field, call the finite field object. For example, `field(0b1101)` will construct the element whose polynomial representation is $X^3 + X^2 + 1$.

    Examples:

        >>> field = komm.FiniteBifield(4)
        >>> x = field(0b1011)
        >>> y = field(0b1100)
        >>> x + y
        0b111
        >>> x * y
        0b1101
        >>> x / y
        0b10
    """

    def __init__(self, degree, modulus=None):
        r"""
        Constructor for the class.

        Parameters:

            degree (int): Degree $k$ of the finite field. Must be a positive integer.

            modulus (Optional[BinaryPolynomial | int]): Modulus (primitive polynomial) $p(X)$ of the field, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former. Must be an irreducible polynomial. If not specified, the modulus is chosen from the table below <cite>LC04, p.42</cite>.

                | Degree $k$ | Modulus $p(X)$ | Degree $k$ | Modulus $p(X)$        |
                | :--------: | -------------- | :--------: | --------------------- |
                | $1$        | `0b11`         | $9$        | `0b1000010001`        |
                | $2$        | `0b111`        | $10$       | `0b10000001001`       |
                | $3$        | `0b1011`       | $11$       | `0b100000000101`      |
                | $4$        | `0b10011`      | $12$       | `0b1000001010011`     |
                | $5$        | `0b100101`     | $13$       | `0b10000000011011`    |
                | $6$        | `0b1000011`    | $14$       | `0b100010001000011`   |
                | $7$        | `0b10001001`   | $15$       | `0b1000000000000011`  |
                | $8$        | `0b100011101`  | $16$       | `0b10001000000001011` |

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

        """
        self._characteristic = 2
        self._degree = degree
        if modulus is None:
            self._modulus = BinaryPolynomial(self._default_modulus(degree))
        else:
            self._modulus = BinaryPolynomial(modulus)

        assert self._modulus.degree == self._degree

    @property
    def characteristic(self):
        r"""
        The characteristic $2$ of the finite field.
        """
        return self._characteristic

    @property
    def degree(self):
        r"""
        The degree $k$ of the finite field.
        """
        return self._degree

    @property
    def modulus(self):
        r"""
        The modulus (primitive polynomial) $p(X)$ of the finite field.
        """
        return self._modulus

    @property
    def order(self):
        r"""
        The order (number of elements) of the finite field. It is given by $2^k$.
        """
        return 2**self._degree

    @property
    def primitive_element(self):
        r"""
        A primitive element $\alpha$ of the finite field. It satisfies $p(\alpha) = 0$, where $p(X)$ is the modulus (primitive polynomial) of the finite field.

        Examples:

            >>> field1 = komm.FiniteBifield(3, modulus=0b1011)
            >>> alpha1 = field1.primitive_element
            >>> [alpha1**i for i in range(7)]
            [0b1, 0b10, 0b100, 0b11, 0b110, 0b111, 0b101]
            >>> field2 = komm.FiniteBifield(3, modulus=0b1101)
            >>> alpha2 = field2.primitive_element
            >>> [alpha2**i for i in range(7)]
            [0b1, 0b10, 0b100, 0b101, 0b111, 0b11, 0b110]
        """
        return self(2)

    # ~@functools.lru_cache(maxsize=None)
    def _multiply(self, x, y):
        return self((BinaryPolynomial(x) * BinaryPolynomial(y)) % self._modulus)

    # ~@functools.lru_cache(maxsize=None)
    def inverse(self, x):
        r"""
        Returns the multiplicative inverse of a given element.
        """
        d, s, _ = BinaryPolynomial.xgcd(BinaryPolynomial(x), self._modulus)
        if d._integer == 1:
            return self(s)
        else:
            raise ZeroDivisionError(
                "This element does not have a multiplicative inverse"
            )

    # ~@functools.lru_cache(maxsize=None)
    def logarithm(self, x, base=None):
        r"""
        Returns the logarithm of a given element, with respect to a given base. If no base is given, the primitive element is used as the base.
        """
        if base is None:
            base = self.primitive_element
        for i in range(self.order):
            if base**i == x:
                return i
        return -1

    def power(self, x, exponent):
        r"""
        Returns a given power of a given element.
        """
        if exponent < 0:
            return power(self.inverse(x), -exponent, self)
        else:
            return power(x, exponent, self)

    @staticmethod
    def conjugates(x):
        r"""
        Returns the conjugates of a given element. See <cite>LC04, Sec. 2.5</cite>.
        """
        conjugate_list = []
        exponent = 0
        while True:
            y = x ** (2**exponent)
            if y not in conjugate_list:
                conjugate_list.append(y)
            else:
                break
            exponent += 1
        return conjugate_list

    @staticmethod
    def minimal_polynomial(x):
        r"""
        Returns the minimal polynomial of a given element. See <cite>LC04, Sec. 2.5</cite>.
        """
        one = x.field(1)
        monomials = [np.array([y, one], dtype=object) for y in x.conjugates()]
        coefficients = functools.reduce(np.convolve, monomials)
        return BinaryPolynomial.from_coefficients(int(c) for c in coefficients)

    def __repr__(self):
        if self._modulus._integer == self._default_modulus(self._degree):
            args = f"{self._degree}"
        else:
            ## modulus must be in binary form
            args = f"{self._degree}, modulus=0b{self._modulus._integer:b}"
        return f"{self.__class__.__name__}({args})"

    def __call__(self, value):
        r"""
        Constructs elements of the finite field.
        """
        element = self._Element(value)
        element.field = self
        return element

    @staticmethod
    def _default_modulus(degree):
        return {
            1: BinaryPolynomial(0b11),
            2: BinaryPolynomial(0b111),
            3: BinaryPolynomial(0b1011),
            4: BinaryPolynomial(0b10011),
            5: BinaryPolynomial(0b100101),
            6: BinaryPolynomial(0b1000011),
            7: BinaryPolynomial(0b10001001),
            8: BinaryPolynomial(0b100011101),
            9: BinaryPolynomial(0b1000010001),
            10: BinaryPolynomial(0b10000001001),
            11: BinaryPolynomial(0b100000000101),
            12: BinaryPolynomial(0b1000001010011),
            13: BinaryPolynomial(0b10000000011011),
            14: BinaryPolynomial(0b100010001000011),
            15: BinaryPolynomial(0b1000000000000011),
            16: BinaryPolynomial(0b10000000010000011),
        }[degree]

    class _Element(int):
        r"""
        Elements of a FiniteBifield.

        Objects of this class represents *elements* of the finite field $\mathrm{GF}(2^k)$.
        """

        field: "FiniteBifield"

        def __eq__(self, other):
            return (
                int(self) == int(other) and self.field._modulus == other.field._modulus
            )

        def __hash__(self):
            return hash((int(self), self.field._modulus))

        def __add__(self, other):
            return self.field(self ^ other)

        def __sub__(self, other):
            return self.field(self ^ other)

        def __mul__(self, other):
            return self.field._multiply(self, other)

        def inverse(self):
            return self.field.inverse(self)

        def __truediv__(self, other):
            return self * other.inverse()

        def logarithm(self, base=None):
            return self.field.logarithm(self, base)

        def __pow__(self, exponent):
            return self.field.power(self, exponent)

        def conjugates(self):
            return self.field.conjugates(self)

        def minimal_polynomial(self):
            return self.field.minimal_polynomial(self)

        def __repr__(self):
            return bin(self)

        def __str__(self):
            return bin(self)
