import numpy as np
import pytest

import komm

from .util import extended_pmf, random_pmf


@pytest.mark.parametrize(
    "assignment, codewords",
    [
        ("cumulative", [(0, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]),
        ("canonical", [(0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]),
    ],
)
def test_shannon_code_wikipedia_1(assignment, codewords):
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.ShannonCode(pmf, assignment)
    assert code.codewords == codewords
    np.testing.assert_allclose(code.average_length(pmf), 102 / 39)


def test_shannon_code_wikipedia_2():
    pmf = [0.36, 0.18, 0.18, 0.12, 0.09, 0.07]
    code = komm.ShannonCode(pmf, assignment="canonical")
    assert code.enc_mapping == {
        0: (0, 0),
        1: (0, 1, 0),
        2: (0, 1, 1),
        3: (1, 0, 0, 0),
        4: (1, 0, 0, 1),
        5: (1, 0, 1, 0),
    }


@pytest.mark.parametrize(
    "counts, codewords",
    [
        ([13, 4, 4, 3], [(0,), (1, 0, 0), (1, 0, 1), (1, 1, 1)]),
        ([16, 5, 3], [(0,), (1, 0, 1), (1, 1, 1)]),
    ],
)
def test_shannon_code_binary_boundary(counts, codewords):
    # Sums of the floats fall below 7/8.
    pmf = np.array(counts) / np.sum(counts)
    code = komm.ShannonCode(pmf)
    assert code.codewords == codewords


@pytest.mark.parametrize(
    "pmf, message",
    [
        ([0.5, 0.5, 0.1], "'pmf' must sum to 1.0"),
        ([0.5, 0.5, 1e-6], "'pmf' must sum to 1.0"),
        ([[0.5], [0.5]], "'pmf' must be a 1D-array"),
        ([1.0], "'pmf' must have size at least 2"),
    ],
)
def test_shannon_code_invalid_pmf(pmf, message):
    with pytest.raises(ValueError, match=message):
        komm.ShannonCode(pmf)


def test_shannon_code_invalid_assignment():
    with pytest.raises(
        ValueError, match="'assignment' must be 'cumulative' or 'canonical'"
    ):
        komm.ShannonCode([0.5, 0.5], assignment="unknown")  # type: ignore


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


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("order", range(1, 4))
def test_shannon_code_assignment(source_cardinality, order, rng):
    # Assignment preserves the lengths, and 'canonical' agrees with 'from_lengths'.
    pmf = extended_pmf(random_pmf(rng, source_cardinality), order)
    code_cumulative = komm.ShannonCode(pmf)
    code_canonical = komm.ShannonCode(pmf, "canonical")
    assert code_canonical.lengths == code_cumulative.lengths
    assert np.isclose(
        code_canonical.average_length(pmf), code_cumulative.average_length(pmf)
    )
    expected = komm.SymbolCode.from_lengths(code_cumulative.lengths)
    assert code_canonical.enc_mapping == expected.enc_mapping
