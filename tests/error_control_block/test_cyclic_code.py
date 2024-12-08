import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "length, generator_polynomial, check_polynomial",
    [
        (7, 0b1011, 0b10111),  # Hamming (7, 4)
        (23, 0b110001110101, 0b1111100100101),  # Golay (23, 12)
    ],
)
def test_cyclic_code(length, generator_polynomial, check_polynomial):
    code_g = komm.CyclicCode(length=length, generator_polynomial=generator_polynomial)
    code_h = komm.CyclicCode(length=length, check_polynomial=check_polynomial)
    assert code_g.check_polynomial == check_polynomial
    assert code_h.generator_polynomial == generator_polynomial


def test_cyclic_code_non_systematic_encode():
    # [LC04, Example 5.1]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=False)
    u = [1, 0, 1, 0]
    v = [1, 1, 1, 0, 0, 1, 0]
    np.testing.assert_array_equal(code.enc_mapping(u), v)
    np.testing.assert_array_equal(code.inv_enc_mapping(v), u)


def test_cyclic_code_systematic_encode():
    # [LC04, Example 5.2]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=True)
    u = [1, 0, 0, 1]
    v = [0, 1, 1, 1, 0, 0, 1]
    np.testing.assert_array_equal(code.enc_mapping(u), v)
    np.testing.assert_array_equal(code.inv_enc_mapping(v), u)


def test_cyclic_code_non_systematic_generator_matrix():
    # [LC04, p. 143-144]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=False)
    generator_matrix = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
    ]
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)


def test_cyclic_code_systematic_generator_matrix():
    # [LC04, p. 143-144]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=True)
    generator_matrix = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
    ]
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)


def test_cyclic_code_check():
    # [LC04, Example 5.7]
    code = komm.CyclicCode(length=7, check_polynomial=0b10111, systematic=True)
    r = [0, 0, 1, 0, 1, 1, 0]
    s = [1, 0, 1]
    np.testing.assert_array_equal(code.chk_mapping(r), s)


def test_cyclic_code_syndrome():
    # [LC04, Example 5.9]
    code = komm.CyclicCode(length=7, check_polynomial=0b10111, systematic=True)
    np.testing.assert_array_equal(code.chk_mapping([0, 0, 0, 0, 0, 0, 1]), [1, 0, 1])
    np.testing.assert_array_equal(code.chk_mapping([0, 0, 0, 0, 0, 1, 0]), [1, 1, 1])
    np.testing.assert_array_equal(code.chk_mapping([0, 0, 0, 0, 1, 0, 0]), [0, 1, 1])
    np.testing.assert_array_equal(code.chk_mapping([0, 0, 0, 1, 0, 0, 0]), [1, 1, 0])
    np.testing.assert_array_equal(code.chk_mapping([0, 0, 1, 0, 0, 0, 0]), [0, 0, 1])
    np.testing.assert_array_equal(code.chk_mapping([0, 1, 0, 0, 0, 0, 0]), [0, 1, 0])
    np.testing.assert_array_equal(code.chk_mapping([1, 0, 0, 0, 0, 0, 0]), [1, 0, 0])


@pytest.mark.parametrize(
    "length, check_polynomial",
    [(7, 0b10111), (23, 0b1111100100101)],
)
@pytest.mark.parametrize(
    "systematic",
    [False, True],
)
def test_cyclic_code_mappings(length, check_polynomial, systematic):
    code = komm.CyclicCode(
        length=length, check_polynomial=check_polynomial, systematic=systematic
    )
    k, m = code.dimension, code.redundancy
    for _ in range(100):
        u = np.random.randint(0, 2, (3, 4, k))
        v = code.enc_mapping(u)
        np.testing.assert_array_equal(
            code.inv_enc_mapping(v),
            u,
        )
        np.testing.assert_array_equal(
            code.chk_mapping(v),
            np.zeros((3, 4, m)),
        )


def test_cyclic_code_inv_enc_mapping_invalid_input():
    code = komm.CyclicCode(length=7, check_polynomial=0b10111)
    r = np.zeros(code.length)
    code.inv_enc_mapping(r)  # Correct
    with np.testing.assert_raises(ValueError):
        r[0] = 1
        code.inv_enc_mapping(r)  # Incorrect
