import functools

import numpy as np

from . import BinaryPolynomial
from .util import power


class FiniteBifield:
    r"""
    Finite field with binary characteristic. Objects of this class represent a *finite field* :math:`\mathrm{GF}(2^k)` (also known as *Galois field*), with *characteristic* :math:`2` and *degree* :math:`k`.

    To construct *elements* of the finite field, call the finite field object. For example, :code:`field(0b1101)` will construct the element whose polynomial representation is :math:`X^3 + X^2 + 1`.

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

            degree (:obj:`int`): Degree :math:`k` of the finite field. Must be a positive integer.

            modulus (:obj:`BinaryPolynomial` or :obj:`int`, optional): Modulus (primitive polynomial) :math:`p(X)` of the field, specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former. Must be an irreducible polynomial. If not specified, the modulus is chosen from the table below.

        .. admonition:: Default values for the primitive polynomials

            From :cite:`Lin.Costello.04` (p. 42).

            ===========  =====================  ============  ============================
            :math:`k`    :math:`p(X)`           :math:`k`     :math:`p(X)`
            ===========  =====================  ============  ============================
            :math:`1`    :code:`0b11`           :math:`9`     :code:`0b1000010001`
            :math:`2`    :code:`0b111`          :math:`10`    :code:`0b10000001001`
            :math:`3`    :code:`0b1011`         :math:`11`    :code:`0b100000000101`
            :math:`4`    :code:`0b10011`        :math:`12`    :code:`0b1000001010011`
            :math:`5`    :code:`0b100101`       :math:`13`    :code:`0b10000000011011`
            :math:`6`    :code:`0b1000011`      :math:`14`    :code:`0b100010001000011`
            :math:`7`    :code:`0b10001001`     :math:`15`    :code:`0b1000000000000011`
            :math:`8`    :code:`0b100011101`    :math:`16`    :code:`0b10001000000001011`
            ===========  =====================  ============  ============================

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
        The characteristic :math:`2` of the finite field. This property is read-only.
        """
        return self._characteristic

    @property
    def degree(self):
        r"""
        The degree :math:`k` of the finite field. This property is read-only.
        """
        return self._degree

    @property
    def modulus(self):
        r"""
        The modulus (primitive polynomial) :math:`p(X)` of the finite field. This property is read-only.
        """
        return self._modulus

    @property
    def order(self):
        r"""
        The order (number of elements) of the finite field. It is given by :math:`2^k`. This property is read-only.
        """
        return 2**self._degree

    @property
    def primitive_element(self):
        r"""
        A primitive element :math:`\alpha` of the finite field. It satisfies :math:`p(\alpha) = 0`, where :math:`p(X)` is the modulus (primitive polynomial) of the finite field. This property is read-only.

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
            raise ZeroDivisionError("This element does not have a multiplicative inverse")

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
        Returns the conjugates of a given element. See :cite:`Lin.Costello.04` (Sec. 2.5).
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
        Returns the minimal polynomial of a given element. See :cite:`Lin.Costello.04` (Sec. 2.5) fore more details.
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

        Objects of this class represents *elements* of the finite field :math:`\mathrm{GF}(2^k)`.
        """
        field: "FiniteBifield"

        def __eq__(self, other):
            return int(self) == int(other) and self.field._modulus == other.field._modulus

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
