import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.HammingCode(3),
        komm.RepetitionCode(5),
        komm.SingleParityCheckCode(5),
        komm.CordaroWagnerCode(5),
        komm.ReedMullerCode(1, 4),
    ]
)
def code(request: pytest.FixtureRequest) -> komm.abc.BlockCode:
    return request.param


def test_gaussian_elimination_compatible(code: komm.abc.BlockCode):
    # The decoded codeword must agree on the unerased positions.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.5)
    decoder = komm.GaussianEliminationDecoder(code)
    for _ in range(100):
        v = code.encode(dms.emit(code.dimension))
        r = bec.transmit(v)
        known = r != 2
        v_hat = code.encode(decoder.decode(r))
        np.testing.assert_equal(v_hat[known], v[known])


def test_gaussian_elimination_unique(code: komm.abc.BlockCode):
    # Fewer erasures than the minimum distance means unique decoding.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.3)
    decoder = komm.GaussianEliminationDecoder(code)
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        if np.count_nonzero(r == 2) < code.minimum_distance():
            np.testing.assert_equal(decoder.decode(r), u)


def test_gaussian_elimination_no_erasures(code: komm.abc.BlockCode):
    # Without erasures, every message is recovered.
    decoder = komm.GaussianEliminationDecoder(code)
    k = code.dimension
    messages = komm.int_to_bits(range(2**k), width=k)
    np.testing.assert_equal(decoder.decode(code.codewords()), messages)
