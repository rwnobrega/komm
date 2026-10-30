import numpy as np
import pytest

import komm


@pytest.mark.parametrize("rho", range(5))
def test_reed_decoder(rho, rng):
    code = komm.ReedMullerCode(5, rho)
    n = code.length  # 32
    k = code.dimension
    t = code.packing_radius()
    decoder = komm.ReedDecoder(code)
    for w in range(t + 1):
        for _ in range(100):
            u = rng.integers(2, size=k)
            r = code.encode(u)
            error_locations = rng.choice(n, w, replace=False)
            r[error_locations] ^= 1
            np.testing.assert_equal(decoder.decode(r), u)


def test_reed_repetition_code(rng):
    code = komm.ReedMullerCode(5, 0)
    assert code.dimension == 1
    assert code.length == 32
    decoder = komm.ReedDecoder(code)
    r = rng.integers(2, size=(100, 32))
    votes = r.sum(axis=1)
    np.testing.assert_equal(
        decoder.decode(r),
        np.where(votes == 16, r[:, 0], votes > 16).reshape(-1, 1),
    )


@pytest.mark.parametrize("rho", range(5))
@pytest.mark.parametrize("input_type", ["hard", "soft"])
def test_reed_codeword_shift(rho, input_type, rng):
    # Adding a codeword adds its message
    code = komm.ReedMullerCode(5, rho)
    decoder = komm.ReedDecoder(code, input_type=input_type)
    u = rng.integers(2, size=(1000, code.dimension))
    r = rng.integers(2, size=(1000, code.length))
    r_shift = (r + code.encode(u)) % 2
    if input_type == "soft":  # Equal magnitudes, as in the BSC
        r, r_shift = (-1.0) ** r, (-1.0) ** r_shift
    np.testing.assert_equal(decoder.decode(r_shift), (decoder.decode(r) + u) % 2)
