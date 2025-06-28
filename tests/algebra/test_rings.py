import pytest

import komm
from komm._algebra import ring


@pytest.mark.parametrize(
    "matrix, determinant",
    [
        ([[0b1, 0b10], [0b10, 0b11]], 0b111),
        ([[0b1, 0b1], [0b10, 0b101]], 0b111),
        ([[0b10, 0b1], [0b11, 0b101]], 0b1001),
    ],
)
def test_ring_determinant_binary_polynomial(matrix, determinant):
    # [McE98, p. 1104]
    matrix = [[komm.BinaryPolynomial(x) for x in row] for row in matrix]
    assert ring.determinant(matrix) == komm.BinaryPolynomial(determinant)
