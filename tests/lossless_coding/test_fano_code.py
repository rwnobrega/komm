import numpy as np
import pytest

import komm


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
