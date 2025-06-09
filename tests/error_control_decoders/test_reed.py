import numpy as np
import pytest

import komm


@pytest.mark.parametrize("mu", range(5))
def test_reed_decoder(mu):
    code = komm.ReedMullerCode(mu, 5)
    n = code.length  # 32
    k = code.dimension
    t = code.packing_radius()
    decoder = komm.ReedDecoder(code)
    for w in range(t + 1):
        for _ in range(100):
            u = np.random.randint(2, size=k)
            r = code.encode(u)
            error_locations = np.random.choice(n, w, replace=False)
            r[error_locations] ^= 1
            np.testing.assert_equal(decoder(r), u)


def test_reed_repetition_code():
    code = komm.ReedMullerCode(0, 5)
    assert code.dimension == 1
    assert code.length == 32
    decoder = komm.ReedDecoder(code)
    r = np.random.randint(2, size=(100, 32))
    np.testing.assert_equal(
        decoder(r),
        (r.sum(axis=1) > 16).astype(int).reshape(-1, 1),
    )
