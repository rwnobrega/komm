import pytest

import komm


def test_simple():
    """
    Lin--Costello, Example 2.7,  p. 46
    """
    field = komm.BinaryFiniteExtensionField(4, 0b10011)
    alpha = field.primitive_element
    one = field(1)
    assert alpha**4 == one + alpha == field(0b0011)
    assert alpha**5 == alpha + alpha**2 == field(0b0110)
    assert alpha**6 == alpha**2 + alpha**3 == field(0b1100)
    assert alpha**7 == one + alpha + alpha**3 == alpha**4 / alpha**12 == \
                       alpha**12 / alpha**5 == field(0b1011)
    assert alpha**13 == alpha**5 + alpha**7 == field(0b1101)
    assert one + alpha**5 + alpha**10 == field(0)

def test_conjugates():
    """
    Lin--Costello, Table 2.9,  p. 52
    """
    field = komm.BinaryFiniteExtensionField(4, 0b10011)
    alpha = field.primitive_element
    assert set(field(0).conjugates()) == {field(0)}
    assert set(field(1).conjugates()) == {field(1)}
    assert set(field(alpha).conjugates()) == {alpha, alpha**2, alpha**4, alpha**8}
    assert set(field(alpha**3).conjugates()) == {alpha**3, alpha**6, alpha**9, alpha**12}
    assert set(field(alpha**5).conjugates()) == {alpha**5, alpha**10}
    assert set(field(alpha**7).conjugates()) == {alpha**7, alpha**11, alpha**13, alpha**14}

def test_minimal_polynomial():
    """
    Lin--Costello, Table 2.9,  p. 52
    """
    field = komm.BinaryFiniteExtensionField(4, 0b10011)
    alpha = field.primitive_element
    assert field(0).minimal_polynomial() == komm.BinaryPolynomial(0b10)
    assert field(1).minimal_polynomial() == komm.BinaryPolynomial(0b11)
    assert field(alpha).minimal_polynomial() == komm.BinaryPolynomial(0b10011)
    assert field(alpha**3).minimal_polynomial() == komm.BinaryPolynomial(0b11111)
    assert field(alpha**5).minimal_polynomial() == komm.BinaryPolynomial(0b111)
    assert field(alpha**7).minimal_polynomial() == komm.BinaryPolynomial(0b11001)


@pytest.mark.parametrize('m', list(range(2, 8)))
def test_inverse(m):
    field = komm.BinaryFiniteExtensionField(m)
    for i in range(1, field.order):
        a = field(i)
        assert a * a.inverse() == field(1)

@pytest.mark.parametrize('m', list(range(2, 8)))
def test_logarithm(m):
    field = komm.BinaryFiniteExtensionField(m)
    alpha = field.primitive_element
    for i in range(1, field.order - 1):
        assert (alpha**i).logarithm() == i
