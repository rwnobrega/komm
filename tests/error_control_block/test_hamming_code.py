import numpy as np
import pytest
from typeguard import TypeCheckError

import komm


def test_hamming_code_parameters():
    code = komm.HammingCode(3)
    assert (code.length, code.dimension, code.redundancy) == (7, 4, 3)
    assert code.minimum_distance() == 3


def test_hamming_code_generator_matrix():
    code = komm.HammingCode(3)
    np.testing.assert_array_equal(
        code.generator_matrix,
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
    )


def test_hamming_code_parity_check_matrix():
    code = komm.HammingCode(3)
    np.testing.assert_array_equal(
        code.check_matrix,
        [
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ],
    )


def test_hamming_code_weight_distributions():
    code = komm.HammingCode(3)
    np.testing.assert_array_equal(
        code.codeword_weight_distribution(),
        [1, 0, 0, 7, 7, 0, 0, 1],
    )
    np.testing.assert_array_equal(
        code.coset_leader_weight_distribution(),
        [1, 7, 0, 0, 0, 0, 0, 0],
    )


def test_hamming_code_GH_orthogonality():
    code = komm.HammingCode(3)
    np.testing.assert_array_equal(
        np.dot(code.generator_matrix, code.check_matrix.T) % 2,
        np.zeros((code.dimension, code.redundancy), dtype=int),
    )


@pytest.mark.parametrize(
    "u, v",
    [
        ([1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]),
        ([1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0]),
        ([1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_hamming_code_encode(u, v):
    code = komm.HammingCode(3)
    np.testing.assert_array_equal(code.encode(u), v)
    np.testing.assert_array_equal(code.inverse_encode(v), u)
    np.testing.assert_array_equal(
        code.encode([[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]]),
        [[1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1]],
    )


def test_hamming_code_codewords():
    code = komm.HammingCode(3)
    assert code.codewords().shape == (16, 7)


def test_extended_hamming_code_parameters():
    code = komm.HammingCode(3, extended=True)
    assert (code.length, code.dimension, code.redundancy) == (8, 4, 4)
    assert code.minimum_distance() == 4


def test_extended_hamming_code_generator_matrix():
    code = komm.HammingCode(3, extended=True)
    np.testing.assert_array_equal(
        code.generator_matrix,
        [
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0],
        ],
    )


def test_extended_hamming_code_parity_check_matrix():
    code = komm.HammingCode(3, extended=True)
    np.testing.assert_array_equal(
        code.check_matrix,
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1],
        ],
    )


def test_extended_hamming_code_weight_distributions():
    code = komm.HammingCode(3, extended=True)
    np.testing.assert_array_equal(
        code.codeword_weight_distribution(),
        [1, 0, 0, 0, 14, 0, 0, 0, 1],
    )
    np.testing.assert_array_equal(
        code.coset_leader_weight_distribution(),
        [1, 8, 7, 0, 0, 0, 0, 0, 0],
    )


def test_extended_hamming_code_GH_orthogonality():
    code = komm.HammingCode(3, extended=True)
    np.testing.assert_array_equal(
        np.dot(code.generator_matrix, code.check_matrix.T) % 2,
        np.zeros((code.dimension, code.redundancy), dtype=int),
    )


@pytest.mark.parametrize(
    "u, v",
    [
        ([1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1, 1]),
        ([1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0]),
        ([1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_extended_hamming_code_encode(u, v):
    code = komm.HammingCode(3, extended=True)
    np.testing.assert_array_equal(code.encode(u), v)
    np.testing.assert_array_equal(code.inverse_encode(v), u)
    np.testing.assert_array_equal(
        code.encode([[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]]),
        [[1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
    )


def test_extended_hamming_code_codewords():
    code = komm.HammingCode(3, extended=True)
    assert code.codewords().shape == (16, 8)


def test_hamming_code_invalid_init():
    with pytest.raises(ValueError, match="'mu' must be at least 2"):
        komm.HammingCode(1)
    with pytest.raises(TypeCheckError):
        komm.HammingCode(7 / 4)  # type: ignore
    with pytest.raises(TypeCheckError):
        komm.HammingCode(7, 4)  # type: ignore
