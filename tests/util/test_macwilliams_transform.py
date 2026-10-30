from math import comb

import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "code",
    [
        komm.BlockCode(generator_matrix=[[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1]]),
        komm.HammingCode(3),
        komm.GolayCode(),
        komm.BCHCode(mu=4, delta=5),
        komm.ReedSolomonCode(mu=3, delta=5),
    ],
)
def test_macwilliams_transform_enumeration(code):
    # Both distributions by enumeration.
    dual = komm.BlockCode(generator_matrix=code.check_matrix)
    minlength = code.length + 1
    A = np.bincount(np.sum(code.codewords(), axis=1), minlength=minlength)
    B = np.bincount(np.sum(dual.codewords(), axis=1), minlength=minlength)
    assert komm.macwilliams_transform(A) == B.tolist()
    assert komm.macwilliams_transform(B) == A.tolist()


@pytest.mark.parametrize("n", [1, 2, 7, 100])
def test_macwilliams_transform_repetition(n):
    # Dual: single parity-check code.
    repetition = [1] + [0] * (n - 1) + [1]
    expected = [comb(n, w) if w % 2 == 0 else 0 for w in range(n + 1)]
    assert komm.macwilliams_transform(repetition) == expected


@pytest.mark.parametrize(
    "distribution, message",
    [
        ([], "must be non-negative and nonzero"),
        ([0, 0, 0], "must be non-negative and nonzero"),
        ([1, -1, 1], "must be non-negative and nonzero"),
        ([1, 1, 1], "must come from a linear code"),  # Fractional transform
        ([1, 0, 3], "must come from a linear code"),  # Negative transform
    ],
)
def test_macwilliams_transform_invalid(distribution, message):
    with pytest.raises(ValueError, match=message):
        komm.macwilliams_transform(distribution)
