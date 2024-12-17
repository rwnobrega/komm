import numpy as np
import pytest

import komm
from komm._algebra.BinaryPolynomial import (
    BinaryPolynomials,
    default_primitive_polynomial,
)
from komm._algebra.domain import DomainElement
from komm._algebra.ring import Ring, RingElement


def test_binary_polynomial_protocol():
    assert isinstance(BinaryPolynomials, Ring)

    poly = komm.BinaryPolynomial(0b10100110111)
    assert isinstance(poly.ambient, BinaryPolynomials)
    assert isinstance(poly, RingElement)
    assert isinstance(poly, DomainElement)


def test_binary_polynomial_hash_and_equality():
    poly0 = komm.BinaryPolynomial(0b10100110111)
    poly1 = komm.BinaryPolynomial(0b10100110111)
    poly2 = komm.BinaryPolynomial(0b101001101110)
    assert poly0 == poly1
    assert poly0 != poly2
    assert hash(poly0) == hash(poly1)
    assert hash(poly0) != hash(poly2)
    assert poly0 == 0b10100110111
    assert poly0 != 0b101001101110
    assert poly0 != "0b10100110111"


def test_binary_polynomial_degree():
    assert komm.BinaryPolynomial(0b10100110111).degree == 10
    assert komm.BinaryPolynomial(0).degree == -1
    assert komm.BinaryPolynomial(1).degree == 0


def test_binary_polynomial_alternative_constructors():
    poly0 = komm.BinaryPolynomial(0b10100110111)
    poly1 = komm.BinaryPolynomial.from_coefficients([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    poly2 = komm.BinaryPolynomial.from_exponents([0, 1, 2, 4, 5, 8, 10])
    assert poly0 == poly1
    assert poly0 == poly2


def test_binary_polynomial_coefficients_and_exponents():
    poly = komm.BinaryPolynomial(0b10100110111)
    assert np.array_equal(poly.coefficients(), [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    assert np.array_equal(poly.exponents(), [0, 1, 2, 4, 5, 8, 10])

    poly = komm.BinaryPolynomial(0b1101)
    assert np.array_equal(poly.coefficients(width=6), [1, 0, 1, 1, 0, 0])

    zero = komm.BinaryPolynomial(0)
    assert np.array_equal(zero.coefficients(), [0])
    assert np.array_equal(zero.exponents(), [])


def test_binary_polynomial_shifts():
    poly = komm.BinaryPolynomial(0b10100110111)
    assert poly >> 2 == komm.BinaryPolynomial(0b101001101)
    assert poly << 2 == komm.BinaryPolynomial(0b1010011011100)
    assert poly << 0 == poly
    assert poly >> 0 == poly

    # shifts that eliminate all terms
    assert poly >> poly.degree == komm.BinaryPolynomial(1)
    assert poly >> (poly.degree + 1) == komm.BinaryPolynomial(0)


def test_binary_polynomial_power():
    poly = komm.BinaryPolynomial(0b10100110111)
    with pytest.raises(ValueError):
        _ = poly ** (-1)
    assert poly**0 == komm.BinaryPolynomial(1)
    assert poly**1 == komm.BinaryPolynomial(0b10100110111)
    assert poly**2 == komm.BinaryPolynomial(0b100010000010100010101)


def test_binary_polynomial_evaluation():
    poly = komm.BinaryPolynomial(0b10100110111)
    assert poly.evaluate(komm.Integer(0)) == komm.Integer(1)
    assert poly.evaluate(komm.Integer(2)) == komm.Integer(0b10100110111)
    assert poly.evaluate(komm.Integer(10)) == komm.Integer(10100110111)
    assert poly.evaluate(komm.Integer(16)) == komm.Integer(0x10100110111)

    # evaluation of zero polynomial
    zero_poly = komm.BinaryPolynomial(0)
    assert zero_poly.evaluate(komm.Integer(42)) == komm.Integer(0)

    # evaluation of constant polynomial
    one_poly = komm.BinaryPolynomial(1)
    assert one_poly.evaluate(komm.Integer(42)) == komm.Integer(1)


def test_binary_polynomial_reciprocal():
    poly = komm.BinaryPolynomial(0b10100110111)
    assert poly.reciprocal() == komm.BinaryPolynomial(0b11101100101)
    np.testing.assert_array_equal(
        poly.reciprocal().coefficients(),
        poly.coefficients()[::-1],
    )


def test_binary_polynomial_arithmetic():
    poly0 = komm.BinaryPolynomial(0b101011)
    poly1 = komm.BinaryPolynomial(0b10011101)
    assert poly0 + poly1 == komm.BinaryPolynomial(0b10110110)
    assert poly0 - poly1 == komm.BinaryPolynomial(0b10110110)
    assert poly0 * poly1 == komm.BinaryPolynomial(0b1011011101111)


def test_binary_polynomial_arithmetic_properties():
    poly = komm.BinaryPolynomial(0b1011)
    zero = komm.BinaryPolynomial(0)
    one = komm.BinaryPolynomial(1)

    assert poly + zero == poly
    assert poly * one == poly
    assert poly + poly == zero  # characteristic 2
    assert -poly == poly  # characteristic 2

    # commutativity
    poly1 = komm.BinaryPolynomial(0b101)
    poly2 = komm.BinaryPolynomial(0b111)
    assert poly1 + poly2 == poly2 + poly1
    assert poly1 * poly2 == poly2 * poly1


def test_binary_polynomial_division():
    poly_dividend = komm.BinaryPolynomial(0b1110011)
    poly_divisor = komm.BinaryPolynomial(0b1011)
    poly_quotient = komm.BinaryPolynomial(0b1100)
    poly_remainder = komm.BinaryPolynomial(0b111)
    assert poly_quotient * poly_divisor + poly_remainder == poly_dividend
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)
    assert poly_dividend // poly_divisor == poly_quotient
    assert poly_dividend % poly_divisor == poly_remainder


def test_binary_polynomial_division_by_zero():
    poly = komm.BinaryPolynomial(0b1011)
    zero = komm.BinaryPolynomial(0)
    with pytest.raises(ZeroDivisionError):
        _ = divmod(poly, zero)
    with pytest.raises(ZeroDivisionError):
        _ = poly // zero
    with pytest.raises(ZeroDivisionError):
        _ = poly % zero


def test_binary_polynomial_division_properties():
    poly = komm.BinaryPolynomial(0b1011)
    assert poly // komm.BinaryPolynomial(1) == poly
    assert poly % komm.BinaryPolynomial(1) == komm.BinaryPolynomial(0)
    assert komm.BinaryPolynomial(0) // poly == komm.BinaryPolynomial(0)
    assert komm.BinaryPolynomial(0) % poly == komm.BinaryPolynomial(0)


def test_binary_polynomial_gcd_lcm():
    poly0 = komm.BinaryPolynomial(0b1101011)
    poly1 = komm.BinaryPolynomial(0b11011)
    poly_gcd = komm.BinaryPolynomial(0b111)
    poly_lcm = komm.BinaryPolynomial(0b111000111)
    assert komm.BinaryPolynomial.gcd(poly0, poly1) == poly_gcd
    assert komm.BinaryPolynomial.lcm(poly0, poly1) == poly_lcm


def test_binary_polynomial_gcd_lcm_multiple():
    poly1 = komm.BinaryPolynomial(0b1100)
    poly2 = komm.BinaryPolynomial(0b1010)
    poly3 = komm.BinaryPolynomial(0b1001)
    gcd = komm.BinaryPolynomial(0b11)
    lcm = komm.BinaryPolynomial(0b101101000)
    assert komm.BinaryPolynomial.gcd(poly1, poly2, poly3) == gcd
    assert komm.BinaryPolynomial.lcm(poly1, poly2, poly3) == lcm


def test_binary_polynomial_gcd_lcm_properties():
    poly = komm.BinaryPolynomial(0b1011)
    zero = komm.BinaryPolynomial(0)
    one = komm.BinaryPolynomial(1)
    assert komm.BinaryPolynomial.gcd(poly, zero) == poly
    assert komm.BinaryPolynomial.gcd(zero, poly) == poly
    assert komm.BinaryPolynomial.gcd(poly, one) == one
    assert komm.BinaryPolynomial.lcm(poly, zero) == zero


def test_binary_polynomial_xgcd():
    poly1 = komm.BinaryPolynomial(0b1101)
    poly2 = komm.BinaryPolynomial(0b111)
    d, s, t = komm.BinaryPolynomial.xgcd(poly1, poly2)
    assert s * poly1 + t * poly2 == d  # Bézout's identity
    assert d == komm.BinaryPolynomial.gcd(poly1, poly2)


def test_binary_polynomial_string_representations():
    poly = komm.BinaryPolynomial(0b101)
    assert str(poly) == "0b101"
    assert repr(poly) == "BinaryPolynomial(0b101)"


def test_binary_polynomial_rabin_irreducibility_test():
    # https://www.ece.unb.ca/tervo/ee4253/polyprime.shtml
    # https://oeis.org/A014580
    # fmt: off
    irreducible = [2, 3, 7, 11, 13, 19, 25, 31, 37, 41, 47, 55, 59, 61, 67, 73, 87, 91, 97, 103, 109, 115, 117, 131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213, 229, 239, 241, 247, 253, 283, 285, 299, 301, 313, 319, 333, 351, 355, 357, 361, 369, 375, 379, 391, 395, 397, 415, 419, 425, 433, 445, 451, 463, 471, 477, 487, 499, 501, 505]
    # fmt: on
    for value in range(513):
        assert komm.BinaryPolynomial(value).is_irreducible() == (value in irreducible)


def test_binary_polynomial_primitive_element():
    # https://www.ece.unb.ca/tervo/ee4253/polyprime.shtml
    # https://oeis.org/A091250
    # fmt: off
    primitive = [3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97, 103, 109, 115, 131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213, 229, 239, 241, 247, 253, 285, 299, 301, 333, 351, 355, 357, 361, 369, 391, 397, 425, 451, 463, 487, 501]
    # fmt: on
    for value in range(513):
        assert komm.BinaryPolynomial(value).is_primitive() == (value in primitive)


def test_binary_polynomial_default_primitive_polynomial():
    for degree in range(1, 25):
        primitive_polynomial = default_primitive_polynomial(degree)
        assert primitive_polynomial.degree == degree
        assert primitive_polynomial.is_irreducible()
        assert primitive_polynomial.is_primitive()
