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


def test_non_systematic_enc_mapping():
    # [LC04, Example 5.1]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=False)
    u = [1, 0, 1, 0]
    v = [1, 1, 1, 0, 0, 1, 0]
    assert np.array_equal(code.enc_mapping(u), v)
    assert np.array_equal(code.inv_enc_mapping(v), u)


def test_systematic_enc_mapping():
    # [LC04, Example 5.2]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=True)
    u = [1, 0, 0, 1]
    v = [0, 1, 1, 1, 0, 0, 1]
    assert np.array_equal(code.enc_mapping(u), v)
    assert np.array_equal(code.inv_enc_mapping(v), u)


def test_non_systematic_generator_matrix():
    # [LC04, p. 143-144]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=False)
    generator_matrix = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
    ]
    assert np.array_equal(code.generator_matrix, generator_matrix)


def test_systematic_generator_matrix():
    # [LC04, p. 143-144]
    code = komm.CyclicCode(length=7, generator_polynomial=0b1011, systematic=True)
    generator_matrix = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
    ]
    assert np.array_equal(code.generator_matrix, generator_matrix)


def test_chk_mapping():
    # [LC04, Example 5.7]
    code = komm.CyclicCode(length=7, check_polynomial=0b10111, systematic=True)
    r = [0, 0, 1, 0, 1, 1, 0]
    s = [1, 0, 1]
    assert np.array_equal(code.chk_mapping(r), s)


def test_syndrome():
    # [LC04, Example 5.9]
    code = komm.CyclicCode(length=7, check_polynomial=0b10111, systematic=True)
    assert np.array_equal(code.chk_mapping([0, 0, 0, 0, 0, 0, 1]), [1, 0, 1])
    assert np.array_equal(code.chk_mapping([0, 0, 0, 0, 0, 1, 0]), [1, 1, 1])
    assert np.array_equal(code.chk_mapping([0, 0, 0, 0, 1, 0, 0]), [0, 1, 1])
    assert np.array_equal(code.chk_mapping([0, 0, 0, 1, 0, 0, 0]), [1, 1, 0])
    assert np.array_equal(code.chk_mapping([0, 0, 1, 0, 0, 0, 0]), [0, 0, 1])
    assert np.array_equal(code.chk_mapping([0, 1, 0, 0, 0, 0, 0]), [0, 1, 0])
    assert np.array_equal(code.chk_mapping([1, 0, 0, 0, 0, 0, 0]), [1, 0, 0])
