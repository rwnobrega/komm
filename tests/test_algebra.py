import pytest

import numpy as np
import komm


def test_binary_polynomial():
    poly = komm.BinaryPolynomial(0b10100110111)
    assert komm.BinaryPolynomial.from_coefficients([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]) == poly
    assert komm.BinaryPolynomial.from_exponents([0, 1, 2, 4, 5, 8, 10]) == poly
    assert poly.degree == 10
    assert np.array_equal(poly.coefficients(), [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    assert np.array_equal(poly.exponents(), [0, 1, 2, 4, 5, 8, 10])
    assert poly == komm.BinaryPolynomial(0b10100110111)
    assert poly >> 2 == komm.BinaryPolynomial(0b101001101)
    assert poly << 2 == komm.BinaryPolynomial(0b1010011011100)
    assert poly ** 2 == komm.BinaryPolynomial(0b100010000010100010101)
    assert poly.evaluate(2) == 0b10100110111
    assert poly.evaluate(10) == 10100110111
    assert poly.evaluate(16) == 0x10100110111

    poly0 = komm.BinaryPolynomial(0b101011)
    poly1 = komm.BinaryPolynomial(0b10011101)
    assert poly0 + poly1 == komm.BinaryPolynomial(0b10110110)
    assert poly0 - poly1 == komm.BinaryPolynomial(0b10110110)
    assert poly0 * poly1 == komm.BinaryPolynomial(0b1011011101111)

    poly_dividend = komm.BinaryPolynomial(0b1110011)
    poly_divisor = komm.BinaryPolynomial(0b1011)
    poly_quotient = komm.BinaryPolynomial(0b1100)
    poly_remainder = komm.BinaryPolynomial(0b111)
    assert poly_quotient * poly_divisor + poly_remainder == poly_dividend
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)
    assert poly_dividend // poly_divisor == poly_quotient
    assert poly_dividend % poly_divisor == poly_remainder

    poly0 = komm.BinaryPolynomial(0b1101011)
    poly1 = komm.BinaryPolynomial(0b11011)
    poly_gcd = komm.BinaryPolynomial(0b111)
    poly_lcm = komm.BinaryPolynomial(0b111000111)

    assert komm.BinaryPolynomial.gcd(poly0, poly1) == poly_gcd
    assert komm.BinaryPolynomial.lcm(poly0, poly1) == poly_lcm


def test_finite_bifield():
    """
    Lin--Costello, Example 2.7,  p. 46.
    """
    field = komm.FiniteBifield(4, 0b10011)
    alpha = field.primitive_element
    one = field(1)
    assert alpha**4 == one + alpha == field(0b0011)
    assert alpha**5 == alpha + alpha**2 == field(0b0110)
    assert alpha**6 == alpha**2 + alpha**3 == field(0b1100)
    assert alpha**7 == one + alpha + alpha**3 == alpha**4 / alpha**12 == alpha**12 / alpha**5 == field(0b1011)
    assert alpha**13 == alpha**5 + alpha**7 == field(0b1101)
    assert one + alpha**5 + alpha**10 == field(0)


def test_conjugates():
    """
    Lin--Costello, Table 2.9,  p. 52.
    """
    field = komm.FiniteBifield(4, 0b10011)
    alpha = field.primitive_element
    assert set(field(0).conjugates()) == {field(0)}
    assert set(field(1).conjugates()) == {field(1)}
    assert set(field(alpha).conjugates()) == {alpha, alpha**2, alpha**4, alpha**8}
    assert set(field(alpha**3).conjugates()) == {alpha**3, alpha**6, alpha**9, alpha**12}
    assert set(field(alpha**5).conjugates()) == {alpha**5, alpha**10}
    assert set(field(alpha**7).conjugates()) == {alpha**7, alpha**11, alpha**13, alpha**14}


def test_minimal_polynomial():
    """
    Lin--Costello, Table 2.9,  p. 52.
    """
    field = komm.FiniteBifield(4, 0b10011)
    alpha = field.primitive_element
    assert field(0).minimal_polynomial() == komm.BinaryPolynomial(0b10)
    assert field(1).minimal_polynomial() == komm.BinaryPolynomial(0b11)
    assert field(alpha).minimal_polynomial() == komm.BinaryPolynomial(0b10011)
    assert field(alpha**3).minimal_polynomial() == komm.BinaryPolynomial(0b11111)
    assert field(alpha**5).minimal_polynomial() == komm.BinaryPolynomial(0b111)
    assert field(alpha**7).minimal_polynomial() == komm.BinaryPolynomial(0b11001)


@pytest.mark.parametrize('m', list(range(2, 8)))
def test_inverse(m):
    field = komm.FiniteBifield(m)
    for i in range(1, field.order):
        a = field(i)
        assert a * a.inverse() == field(1)


@pytest.mark.parametrize('m', list(range(2, 8)))
def test_logarithm(m):
    field = komm.FiniteBifield(m)
    alpha = field.primitive_element
    for i in range(1, field.order - 1):
        assert (alpha**i).logarithm() == i


def test_rational_polynomial():
    assert komm.RationalPolynomial([1, 0, -1]) == komm.RationalPolynomial([1, 0, -1, 0, 0, 0])

    poly0 = komm.RationalPolynomial([1])
    poly1 = komm.RationalPolynomial([0, 1])
    assert poly0 + poly1 == poly1 + poly0 == komm.RationalPolynomial([1, 1])

    poly0 = komm.RationalPolynomial([5, -2, 0, 2, 1, 3])
    poly1 = komm.RationalPolynomial([2, 7, 0, 3, 0, 2])
    assert poly0 + poly1 == komm.RationalPolynomial([7, 5, 0, 5, 1, 5])
    assert poly0 - poly1 == komm.RationalPolynomial([3, -9, 0, -1, 1, 1])

    poly0 = komm.RationalPolynomial([5, 0, 0, 0, 0, 2, 3])
    poly1 = komm.RationalPolynomial([2, 5])
    assert poly0 * poly1 == komm.RationalPolynomial([10, 25, 0, 0, 0, 4, 16, 15])


def test_rational_polynomial_divmod():
    poly_dividend = komm.RationalPolynomial([-4, 0, -2, 1])
    poly_divisor = komm.RationalPolynomial([-3, 1])
    poly_quotient = komm.RationalPolynomial([3, 1, 1])
    poly_remainder = komm.RationalPolynomial([5])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([1, 0, 2])
    poly_divisor = komm.RationalPolynomial([0, 0, 0, 0, 0, 1])
    poly_quotient = komm.RationalPolynomial([])
    poly_remainder = komm.RationalPolynomial([1, 0, 2])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([0, 0, 0, 0, 0, 1])
    poly_divisor = komm.RationalPolynomial([1, 0, 2])
    poly_quotient = komm.RationalPolynomial([0, '-1/4', 0, '1/2'])
    poly_remainder = komm.RationalPolynomial([0, '1/4'])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([12, -26, 21, -9, 2])
    poly_divisor = komm.RationalPolynomial([-3, 2])
    poly_quotient = komm.RationalPolynomial([-4, 6, -3, 1])
    poly_remainder = komm.RationalPolynomial([])
    assert poly_dividend // poly_divisor == poly_quotient
    assert poly_dividend % poly_divisor == poly_remainder


def test_rational_polynomial_gcd():
    poly0 = komm.RationalPolynomial([0, 2, 4])
    poly1 = komm.RationalPolynomial([0, 0, 0, 10])
    poly_gcd = komm.RationalPolynomial([0, 1])
    assert komm.RationalPolynomial.gcd(poly0, poly1) == poly_gcd

    poly0 = komm.RationalPolynomial([6, 7, 1])
    poly1 = komm.RationalPolynomial([-6, -5, 1])
    poly_gcd = komm.RationalPolynomial([1, 1])
    assert komm.RationalPolynomial.gcd(poly0, poly1) == poly_gcd


def test_rational_polynomial_fractions():
    fraction = komm.RationalPolynomialFraction([0, 1, 2], [0, 0, 0, 1])
    assert fraction.numerator == komm.RationalPolynomial([1, 2])
    assert fraction.denominator == komm.RationalPolynomial([0, 0, 1])

    fraction = komm.RationalPolynomialFraction([0, '5/14'], [0, 0, 0, '55/21'])
    assert fraction.numerator == komm.RationalPolynomial([3])
    assert fraction.denominator == komm.RationalPolynomial([0, 0, 22])
