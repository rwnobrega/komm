import numpy as np

import komm


def test_fano_code_wikipedia():
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.FanoCode(pmf)
    assert code.enc_mapping == {
        (0,): (0, 0),
        (1,): (0, 1),
        (2,): (1, 0),
        (3,): (1, 1, 0),
        (4,): (1, 1, 1),
    }
    np.testing.assert_allclose(code.rate(pmf), 89 / 39)
