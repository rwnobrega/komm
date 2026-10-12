import numpy as np
import pytest

import komm

from .util import random_pmf


def test_shannon_code_wikipedia_1():
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.ShannonCode(pmf)
    assert code.enc_mapping == {
        0: (0, 0),
        1: (0, 1, 0),
        2: (0, 1, 1),
        3: (1, 0, 0),
        4: (1, 0, 1),
    }
    np.testing.assert_allclose(code.average_length(pmf), 102 / 39)


def test_shannon_code_wikipedia_2():
    pmf = [0.36, 0.18, 0.18, 0.12, 0.09, 0.07]
    code = komm.ShannonCode(pmf)
    assert code.enc_mapping == {
        0: (0, 0),
        1: (0, 1, 0),
        2: (0, 1, 1),
        3: (1, 0, 0, 0),
        4: (1, 0, 0, 1),
        5: (1, 0, 1, 0),
    }


@pytest.mark.parametrize(
    "pmf, message",
    [
        ([0.5, 0.5, 0.1], "'pmf' must sum to 1.0"),
        ([[0.5], [0.5]], "'pmf' must be a 1D-array"),
        ([1.0], "'pmf' must have size at least 2"),
    ],
)
def test_shannon_code_invalid_pmf(pmf, message):
    with pytest.raises(ValueError, match=message):
        komm.ShannonCode(pmf)


@pytest.mark.parametrize("pmf", [[0.5, 0.5, 0.0], [1.0, 0.0], [0.0, 0.4, 0.6]])
def test_shannon_code_zero_probability(pmf):
    with pytest.raises(ValueError, match="'pmf' must be positive"):
        komm.ShannonCode(pmf)


@pytest.mark.parametrize("S", range(2, 7))
def test_shannon_code_kraft(S, rng):
    # Each 2^-l(x) exceeds p(x) / 2.
    for _ in range(10):
        pmf = random_pmf(rng, S)
        code = komm.ShannonCode(pmf)
        assert 0.5 < code.kraft_parameter() <= 1


def test_shannon_code_almost_deterministic():
    code = komm.ShannonCode([1.0, 1e-17])
    assert code.codewords[0] == (0,)
    assert code.is_prefix_free()
    assert code.kraft_parameter() <= 1
