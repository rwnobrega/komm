from random import randint

import numpy as np
import pytest

import komm
from komm._algebra.field import FieldElement
from komm._algebra.FiniteBifield import horner
from komm._algebra.ring import Ring, RingElement


def test_finite_bifield_protocol():
    field = komm.FiniteBifield(4)
    assert isinstance(field, Ring)

    x = field(0b1011)
    assert isinstance(x.ambient, komm.FiniteBifield)
    assert isinstance(x, RingElement)
    assert isinstance(x, FieldElement)


def test_finite_bifield_constructor():
    field = komm.FiniteBifield(4)
    assert field.degree == 4
    assert field.modulus == komm.BinaryPolynomial(0b10011)  # default modulus

    field = komm.FiniteBifield(4, 0b11001)
    assert field.degree == 4
    assert field.modulus == komm.BinaryPolynomial(0b11001)


def test_finite_bifield_invalid_degree():
    with pytest.raises(ValueError):
        komm.FiniteBifield(0)
    with pytest.raises(ValueError):
        komm.FiniteBifield(-1)


def test_finite_bifield_invalid_modulus():
    komm.FiniteBifield(2, 0b111)  # OK  (irreducible and primitive)
    with pytest.raises(ValueError):
        komm.FiniteBifield(3, 0b111)  # degree mismatch
    with pytest.raises(ValueError):
        komm.FiniteBifield(2, 0b101)  # reducible polynomial
    with pytest.raises(ValueError):
        komm.FiniteBifield(3, 0b101)  # both degree mismatch and reducible polynomial

    komm.FiniteBifield(4, 0b11111)  # OK  (irreducible but not primitive)
    with pytest.raises(ValueError):
        komm.FiniteBifield(3, 0b11111)  # degree mismatch
    with pytest.raises(ValueError):
        komm.FiniteBifield(4, 0b10001)  # reducible polynomial
    with pytest.raises(ValueError):
        komm.FiniteBifield(3, 0b10001)  # both degree mismatch and reducible polynomial


def test_finite_bifield_element_construction():
    field = komm.FiniteBifield(4)
    x1 = field(0b1011)
    x2 = field(komm.BinaryPolynomial(0b1011))
    assert x1 == x2


def test_finite_bifield_different_fields():
    field1 = komm.FiniteBifield(4)
    field2 = komm.FiniteBifield(4, 0b11001)
    x1 = field1(0b1011)
    x2 = field2(0b1011)
    assert x1 != x2
    with pytest.raises(ValueError):
        _ = x1 + x2
    with pytest.raises(ValueError):
        _ = x1 - x2
    with pytest.raises(ValueError):
        _ = x1 * x2
    with pytest.raises(ValueError):
        _ = x1 / x2


def test_finite_bifield_same_field():
    field1 = komm.FiniteBifield(4)
    field2 = komm.FiniteBifield(4)
    field3 = komm.FiniteBifield(4)
    x1 = field1(0b1011)
    x2 = field2(0b1011)
    assert x1 == x2
    assert x1 + x2 == field3(0b0)
    assert x1 - x2 == field3(0b0)
    assert x1 * x2 == field3(0b1001)
    assert x1 / x2 == field3(0b1)


@pytest.mark.parametrize("k", list(range(2, 8)))
def test_finite_bifield_properties(k):
    field = komm.FiniteBifield(k)
    assert field.characteristic == 2
    assert field.order == 2**k
    zero = field.zero
    one = field.one
    assert zero + zero == zero
    assert one * one == one
    assert zero * one == zero


@pytest.mark.parametrize("m", range(2, 8))
def test_finite_bifield_integer_multiplication(m):
    field = komm.FiniteBifield(m)
    for value in range(field.order):
        x = field(value)
        zero = field.zero
        assert 0 * x == zero
        assert 1 * x == x
        assert 2 * x == zero
        assert 3 * x == x
        assert 4 * x == zero
        assert 5 * x == x


@pytest.mark.parametrize("m", range(2, 8))
def test_finite_bifield_element_properties(m):
    field = komm.FiniteBifield(m)
    for value in range(field.order):
        x = field(value)
        zero = field.zero
        one = field.one
        assert x + zero == x
        assert x + x == zero
        assert x - x == zero
        assert x == -x
        assert x * zero == zero
        assert x * one == x


@pytest.mark.parametrize("m", range(2, 8))
def test_finite_bifield_element_field_properties(m):
    field = komm.FiniteBifield(m)
    x = field(randint(0, field.order - 1))
    y = field(randint(0, field.order - 1))
    z = field(randint(0, field.order - 1))

    # Associativity
    assert (x + y) + z == x + (y + z)
    assert (x * y) * z == x * (y * z)

    # Commutativity
    assert x + y == y + x
    assert x * y == y * x

    # Distributivity
    assert x * (y + z) == x * y + x * z

    # Frobenius automorphism in characteristic 2
    assert (x + y) ** 2 == x**2 + y**2


@pytest.mark.parametrize("m", range(2, 8))
def test_finite_bifield_division_and_inverse(m):
    field = komm.FiniteBifield(m)
    for value in range(field.order):
        x = field(value)
        if x != field.zero:
            assert field.zero / x == field.zero
            assert x / x == field.one
            assert x * x.inverse() == field.one
        else:
            with pytest.raises(ZeroDivisionError):
                _ = x / x
            with pytest.raises(ZeroDivisionError):
                _ = x.inverse()


@pytest.mark.parametrize("m", range(2, 8))
def test_finite_bifield_power(m):
    field = komm.FiniteBifield(m)
    for value in range(field.order):
        x = field(value)
        assert x**1 == x
        assert x**2 == x * x
        assert x**3 == x * x * x
        assert x**field.order == x
        if x != field.zero:
            assert x**0 == field.one
            assert x**-1 == x.inverse()
            assert x**-2 == (x * x).inverse()
            assert x ** (field.order - 1) == field.one


@pytest.mark.parametrize("k", list(range(2, 8)))
def test_finite_bifield_logarithm(k):
    field = komm.FiniteBifield(k)
    alpha = field(0b10)
    for i in range(field.order - 1):
        assert (alpha**i).logarithm(alpha) == i


def test_finite_bifield_logarithm_invalid_base():
    field = komm.FiniteBifield(4)
    x = field(0b1011)
    with pytest.raises(ValueError):
        x.logarithm(field.zero)
    with pytest.raises(ValueError):
        x.logarithm(field.one)


@pytest.mark.parametrize("k", range(1, 25))
def test_finite_bifield_primitive_element_default_modulus(k):
    field = komm.FiniteBifield(k)
    assert field.primitive_element == (field.one if k == 1 else field(0b10))


@pytest.mark.parametrize(
    "k, modulus",
    [(4, 0b11111), (6, 0b1001001), (8, 0b100011011)],
)
def test_finite_bifield_primitive_element_non_primitive_modulus(k, modulus):
    field = komm.FiniteBifield(k, modulus)
    alpha = field.primitive_element
    assert alpha == field(0b11)
    assert len({alpha**i for i in range(field.order - 1)}) == field.order - 1


@pytest.mark.parametrize("modulus", [0b10011, 0b11111])
def test_finite_bifield_vectorized_arithmetic(modulus):
    field = komm.FiniteBifield(4, modulus)
    elements = [field(value) for value in range(field.order)]
    x, y = np.meshgrid(range(16), range(16), indexing="ij")
    products = [[int(a * b) for b in elements] for a in elements]
    quotients = [[int(a / b) for b in elements[1:]] for a in elements]
    assert np.array_equal(field.multiply(x, y), products)
    assert np.array_equal(field.divide(x[:, 1:], y[:, 1:]), quotients)
    x, exponent = np.meshgrid(range(1, 16), range(-3, 4), indexing="ij")
    powers = [[int(a**e) for e in range(-3, 4)] for a in elements[1:]]
    assert np.array_equal(field.power(x, exponent), powers)
    assert np.array_equal(field.power(0, [0, 1, 2]), [1, 0, 0])
    with pytest.raises(ZeroDivisionError):
        field.divide(1, 0)
    with pytest.raises(ZeroDivisionError):
        field.power(0, -1)


@pytest.mark.parametrize("modulus", [0b10011, 0b11111])
def test_finite_bifield_horner(modulus):
    def naive_evaluate(field, p: list[int], x: int) -> int:
        return int(sum((field(c) * field(x) ** i for i, c in enumerate(p)), field.zero))

    field = komm.FiniteBifield(4, modulus)
    coefficients = np.random.randint(0, 16, (3, 5))
    points = np.arange(16)
    expected = [
        [naive_evaluate(field, p, x) for x in points.tolist()]
        for p in coefficients.tolist()
    ]
    assert np.array_equal(horner(field, coefficients, points), expected)


def test_finite_bifield_LC_example_2_7():
    """
    Lin--Costello, Example 2.7, p. 46.
    """
    field = komm.FiniteBifield(4, 0b10011)
    alpha = field(0b10)
    one = field.one

    assert alpha**4 == one + alpha == field(0b0011)
    assert alpha**5 == alpha + alpha**2 == field(0b0110)
    assert alpha**6 == alpha**2 + alpha**3 == field(0b1100)
    assert (
        alpha**7
        == one + alpha + alpha**3
        == alpha**4 / alpha**12
        == alpha**12 / alpha**5
        == field(0b1011)
    )
    assert alpha**13 == alpha**5 + alpha**7 == field(0b1101)
    assert one + alpha**5 + alpha**10 == field.zero


def test_finite_bifield_LC_table_2_9():
    r"""
    Lin--Costello, Table 2.9,  p. 52.
    """
    field = komm.FiniteBifield(4, 0b10011)
    alpha = field(0b10)

    assert set(field.zero.conjugates()) == {field.zero}
    assert set(field.one.conjugates()) == {field.one}
    assert set(alpha.conjugates()) == {alpha, alpha**2, alpha**4, alpha**8}
    assert set((alpha**3).conjugates()) == {alpha**3, alpha**6, alpha**9, alpha**12}
    assert set((alpha**5).conjugates()) == {alpha**5, alpha**10}
    assert set((alpha**7).conjugates()) == {alpha**7, alpha**11, alpha**13, alpha**14}

    assert field.zero.minimal_polynomial() == komm.BinaryPolynomial(0b10)
    assert field.one.minimal_polynomial() == komm.BinaryPolynomial(0b11)
    assert alpha.minimal_polynomial() == komm.BinaryPolynomial(0b10011)
    assert (alpha**3).minimal_polynomial() == komm.BinaryPolynomial(0b11111)
    assert (alpha**5).minimal_polynomial() == komm.BinaryPolynomial(0b111)
    assert (alpha**7).minimal_polynomial() == komm.BinaryPolynomial(0b11001)


def test_finite_bifield_string_representation():
    field1 = komm.FiniteBifield(4)  # Default modulus
    assert repr(field1) == "FiniteBifield(4)"
    field2 = komm.FiniteBifield(4, 0b11001)  # Custom modulus
    assert repr(field2) == "FiniteBifield(4, modulus=0b11001)"


def test_finite_bifield_element_string_representation():
    field = komm.FiniteBifield(4)
    assert str(field.zero) == "0b0"
    assert str(field.one) == "0b1"
    x = field(0b1011)
    assert str(x) == "0b1011"
    assert repr(x) == "0b1011"


def test_finite_bifield_element_int():
    field = komm.FiniteBifield(4)
    for value in range(field.order):
        assert int(field(value)) == value
