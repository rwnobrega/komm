import numpy as np
import pytest

import komm
from komm._algebra.bifield import (
    binary_matrix,
    convolve,
    deconvolve,
    divide,
    horner,
    multiply,
    power,
)

params = []

# Default modulus (primitive)
for degree in range(1, 7):
    params.append(komm.FiniteBifield(degree))

# Non-primitive modulus
for degree, modulus in [(4, 0b11111), (6, 0b1001001)]:
    params.append(komm.FiniteBifield(degree, modulus))


@pytest.fixture(params=params, ids=lambda field: repr(field))
def field(request: pytest.FixtureRequest):
    return request.param


def test_bifield_multiply_and_divide(field: komm.FiniteBifield):
    elements = [field(value) for value in range(field.order)]
    x, y = np.meshgrid(range(field.order), range(field.order), indexing="ij")
    assert np.array_equal(
        multiply(field, x, y),
        [[int(a * b) for b in elements] for a in elements],
    )
    assert np.array_equal(
        divide(field, x[:, 1:], y[:, 1:]),
        [[int(a / b) for b in elements[1:]] for a in elements],
    )


def test_bifield_power(field: komm.FiniteBifield):
    elements = [field(value) for value in range(field.order)]
    exponents = range(-field.order, field.order + 1)
    b, e = np.meshgrid(range(1, field.order), exponents, indexing="ij")
    assert np.array_equal(
        power(field, b, e),
        [[int(a**e) for e in exponents] for a in elements[1:]],
    )


def test_bifield_power_zero_base(field: komm.FiniteBifield):
    assert np.array_equal(power(field, 0, [0, 1, 2]), [1, 0, 0])


def test_bifield_division_by_zero(field: komm.FiniteBifield):
    with pytest.raises(ZeroDivisionError):
        divide(field, 1, 0)
    with pytest.raises(ZeroDivisionError):
        power(field, 0, -1)
    with pytest.raises(ZeroDivisionError):
        deconvolve(field, [1, 1], [1, 0])


def test_bifield_horner(field: komm.FiniteBifield):
    def naive_evaluate(p: list[int], x: int) -> int:
        return int(sum((field(c) * field(x) ** i for i, c in enumerate(p)), field.zero))

    coefficients = np.random.randint(0, field.order, (3, 5))
    points = np.arange(field.order)
    expected = [[naive_evaluate(p, x) for x in points.tolist()] for p in coefficients]
    assert np.array_equal(horner(field, coefficients, points), expected)


def test_bifield_convolve(field: komm.FiniteBifield):
    def naive_convolve(p: list[int], q: list[int]) -> list[int]:
        product = [field.zero] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                product[i + j] += field(a) * field(b)
        return [int(c) for c in product]

    x = np.random.randint(0, field.order, (3, 1, 5))
    y = np.random.randint(0, field.order, (4, 3))
    expected = [[naive_convolve(p, q) for q in y.tolist()] for p in x[:, 0].tolist()]
    assert np.array_equal(convolve(field, x, y), expected)


def test_bifield_deconvolve(field: komm.FiniteBifield):
    x = np.random.randint(0, field.order, (3, 1, 7))
    y = np.random.randint(0, field.order, (4, 3))
    y[:, -1] = np.random.randint(1, field.order, 4)
    quotient, remainder = deconvolve(field, x, y)
    assert quotient.shape == (3, 4, 5)
    assert remainder.shape == (3, 4, 2)
    product = convolve(field, quotient, y)
    product[..., :2] ^= remainder
    assert np.array_equal(product, np.broadcast_to(x, (3, 4, 7)))


def test_bifield_binary_matrix(field: komm.FiniteBifield):
    k = field.degree
    matrix = np.random.randint(0, field.order, (3, 5))
    binary = binary_matrix(field, matrix)
    assert binary.shape == (3 * k, 5 * k)
    # Check that φ(a A) = φ(a) B.
    u = np.random.randint(0, 2, (10, 3 * k))
    a = komm.bits_to_int(u, width=k)
    products = multiply(field, a[:, :, np.newaxis], matrix)
    expected = np.bitwise_xor.reduce(products, axis=1)
    assert np.array_equal(komm.bits_to_int(u @ binary % 2, width=k), expected)
