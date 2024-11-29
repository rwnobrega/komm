import pytest

import komm
from komm._algebra.BinaryPolynomialFraction import BinaryPolynomialFractions
from komm._algebra.field import FieldElement
from komm._algebra.ring import Ring, RingElement


def test_binary_polynomial_fraction_protocol():
    assert isinstance(BinaryPolynomialFractions, Ring)

    fraction = komm.BinaryPolynomialFraction(0b1011, 0b1)
    assert isinstance(fraction.ambient, BinaryPolynomialFractions)
    assert isinstance(fraction, RingElement)
    assert isinstance(fraction, FieldElement)


def test_binary_polynomial_fraction_constructor():
    # basic construction
    fraction = komm.BinaryPolynomialFraction(0b1011101, 0b11)
    assert fraction.numerator == komm.BinaryPolynomial(0b1011101)
    assert fraction.denominator == komm.BinaryPolynomial(0b11)

    # default denominator
    fraction = komm.BinaryPolynomialFraction(0b1011101)
    assert fraction.numerator == komm.BinaryPolynomial(0b1011101)
    assert fraction.denominator == komm.BinaryPolynomial(0b1)

    # construction with BinaryPolynomial objects
    num = komm.BinaryPolynomial(0b1011)
    den = komm.BinaryPolynomial(0b1101)
    fraction = komm.BinaryPolynomialFraction(num, den)
    assert fraction.numerator == num
    assert fraction.denominator == den


def test_binary_polynomial_fraction_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        komm.BinaryPolynomialFraction(0b1011, 0b0)


def test_binary_polynomial_fraction_simplification():
    fraction = komm.BinaryPolynomialFraction(0b1100, 0b101)  # (x^3 + x^2) / (x^2 + x)
    assert fraction == komm.BinaryPolynomialFraction(0b100, 0b11)  # (x^2) / (x + 1)


def test_binary_polynomial_fraction_operations():
    p = komm.BinaryPolynomialFraction(0b101101, 0b1010011)
    q = komm.BinaryPolynomialFraction(0b10101011, 0b10101)

    assert -p == komm.BinaryPolynomialFraction(0b101, 0b1011)
    assert p.inverse() == komm.BinaryPolynomialFraction(0b1011, 0b101)
    assert p + q == komm.BinaryPolynomialFraction(0b10011100100, 0b10010111)
    assert p - q == komm.BinaryPolynomialFraction(0b10011100100, 0b10010111)
    assert p * q == komm.BinaryPolynomialFraction(0b1000000111, 0b10010111)
    assert p / q == komm.BinaryPolynomialFraction(0b1000001, 0b10010100101)


def test_binary_polynomial_fraction_power():
    fraction = komm.BinaryPolynomialFraction(0b11, 0b101)  # (x+1)/(x^2+1)

    # Test positive powers
    assert fraction**0 == komm.BinaryPolynomialFraction(1, 1)
    assert fraction**1 == fraction
    assert fraction**2 == fraction * fraction

    # Test negative powers
    assert fraction**-1 == fraction.inverse()
    assert fraction**-2 == (fraction * fraction).inverse()


def test_binary_polynomial_fraction_integer_multiplication():
    fraction = komm.BinaryPolynomialFraction(0b10111001, 0b11010010)
    zero = komm.BinaryPolynomialFraction(0, 1)
    assert 0 * fraction == zero
    assert 1 * fraction == fraction
    assert 2 * fraction == zero
    assert 3 * fraction == fraction
    assert 4 * fraction == zero
    assert 5 * fraction == fraction


def test_binary_polynomial_fractions_field():
    field = BinaryPolynomialFractions()

    # zero and one
    assert field.zero == komm.BinaryPolynomialFraction(0, 1)
    assert field.one == komm.BinaryPolynomialFraction(1, 1)

    # call method
    fraction = field((0b1011, 0b1101))
    assert isinstance(fraction, komm.BinaryPolynomialFraction)
    assert fraction.numerator == komm.BinaryPolynomial(0b1011)
    assert fraction.denominator == komm.BinaryPolynomial(0b1101)


def test_binary_polynomial_fraction_string_representation():
    fraction = komm.BinaryPolynomialFraction(0b1011, 0b1101)
    assert str(fraction) == "0b1011/0b1101"

    fraction = komm.BinaryPolynomialFraction(0b1011, 0b1)
    assert str(fraction) == "0b1011/0b1"

    fraction = komm.BinaryPolynomialFraction(0b1011, 0b1101)
    assert repr(fraction) == "BinaryPolynomialFraction(0b1011, 0b1101)"
