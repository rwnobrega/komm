import numpy as np
import pytest
from typeguard import TypeCheckError

import komm


def test_golay_code_parameters():
    code = komm.GolayCode()
    assert (code.length, code.dimension, code.redundancy) == (23, 12, 11)
    assert code.minimum_distance() == 7


def test_golay_code_codeword_weight_distribution():
    code = komm.GolayCode()
    np.testing.assert_equal(
        code.codeword_weight_distribution(),
        # fmt: off
        [1, 0, 0, 0, 0, 0, 0, 253, 506, 0, 0, 1288, 1288, 0, 0, 506, 253, 0, 0, 0, 0, 0, 0, 1],
        # fmt: on
    )


def test_golay_code_coset_leader_weight_distribution():
    code = komm.GolayCode()
    np.testing.assert_equal(
        code.coset_leader_weight_distribution(),
        # fmt: off
        [1, 23, 253, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # fmt: on
    )


def test_golay_code_GH_orthogonality():
    code = komm.GolayCode()
    np.testing.assert_equal(
        np.dot(code.generator_matrix, code.check_matrix.T) % 2,
        np.zeros((code.dimension, code.redundancy), dtype=int),
    )


def test_golay_code_codewords():
    code = komm.GolayCode()
    assert code.codewords().shape == (4096, 23)


def test_extended_golay_code_parameters():
    code = komm.GolayCode(extended=True)
    assert (code.length, code.dimension, code.redundancy) == (24, 12, 12)
    assert code.minimum_distance() == 8


def test_extended_golay_code_codeword_weight_distribution():
    code = komm.GolayCode(extended=True)
    np.testing.assert_equal(
        code.codeword_weight_distribution(),
        # fmt: off
        [1, 0, 0, 0, 0, 0, 0, 0, 759, 0, 0, 0, 2576, 0, 0, 0, 759, 0, 0, 0, 0, 0, 0, 0, 1],
        # fmt: off
    )


def test_extended_golay_code_coset_leader_weight_distribution():
    code = komm.GolayCode(extended=True)
    np.testing.assert_equal(
        code.coset_leader_weight_distribution(),
        # fmt: off
        [1, 24, 276, 2024, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # fmt: on
    )


def test_extended_golay_code_GH_orthogonality():
    code = komm.GolayCode(extended=True)
    np.testing.assert_equal(
        np.dot(code.generator_matrix, code.check_matrix.T) % 2,
        np.zeros((code.dimension, code.redundancy), dtype=int),
    )


def test_extended_golay_code_codewords():
    code = komm.GolayCode(extended=True)
    assert code.codewords().shape == (4096, 24)


def test_golay_code_invalid_init():
    with pytest.raises(TypeError):
        komm.GolayCode(23, 12)  # type: ignore
    with pytest.raises(TypeCheckError):
        komm.GolayCode(extended=1)  # type: ignore
