import numpy as np
import pytest

import komm


def test_tunstall_code():
    # Sayood.06, p. 71.
    pmf = [0.6, 0.3, 0.1]
    code = komm.TunstallCode(pmf, code_block_size=3)
    assert code.enc_mapping == {
        (0, 0, 0): (0, 0, 0),
        (0, 0, 1): (0, 0, 1),
        (0, 0, 2): (0, 1, 0),
        (0, 1): (0, 1, 1),
        (0, 2): (1, 0, 0),
        (1,): (1, 0, 1),
        (2,): (1, 1, 0),
    }
    assert np.isclose(code.rate(pmf), 3 / 1.96)


def test_tunstall_code_invalid_init():
    with pytest.raises(ValueError):
        komm.TunstallCode([0.5, 0.5, 0.1], code_block_size=3)
    with pytest.raises(ValueError):
        komm.TunstallCode([0.5, 0.5], code_block_size=0)
