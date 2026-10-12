import numpy as np
import pytest

import komm

from .util import extended_pmf, random_pmf


def test_fano_code_wikipedia():
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.FanoCode(pmf)
    assert code.enc_mapping == {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1, 0),
        4: (1, 1, 1),
    }
    np.testing.assert_allclose(code.average_length(pmf), 89 / 39)


@pytest.mark.parametrize(
    "counts, codewords",
    [
        (
            [6, 94],
            [(1,), (0,)],
        ),
        (
            [1, 1, 1],
            [(0, 0), (0, 1), (1,)],
        ),
        (
            [12, 11, 10, 9, 8, 7],
            [(0, 0), (0, 1, 0), (0, 1, 1), (1, 0), (1, 1, 0), (1, 1, 1)],
        ),
    ],
)
def test_fano_code_partition(counts, codewords):
    # Partition codewords, not the canonical ones.
    pmf = np.array(counts) / np.sum(counts)
    code = komm.FanoCode(pmf)
    assert code.codewords == codewords


@pytest.mark.parametrize(
    "pmf, message",
    [
        ([0.5, 0.5, 0.1], "'pmf' must sum to 1.0"),
        ([[0.5], [0.5]], "'pmf' must be a 1D-array"),
        ([1.0], "'pmf' must have size at least 2"),
    ],
)
def test_fano_code_invalid_pmf(pmf, message):
    with pytest.raises(ValueError, match=message):
        komm.FanoCode(pmf)


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("order", range(1, 4))
def test_fano_code_complete(source_cardinality, order, rng):
    # Full binary tree, so Kraft equality holds.
    pmf = extended_pmf(random_pmf(rng, source_cardinality), order)
    code = komm.FanoCode(pmf)
    assert np.isclose(code.kraft_parameter(), 1.0)
