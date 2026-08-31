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


def test_inactivation_agrees_with_gaussian(code: komm.abc.BlockCode):
    # Both optimal decoders return identical output.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.6)
    dec_inactivation = komm.InactivationDecoder(code)
    dec_gaussian = komm.GaussianEliminationDecoder(code)
    for _ in range(100):
        r = bec.transmit(code.encode(dms.emit(code.dimension)))
        np.testing.assert_equal(dec_inactivation.decode(r), dec_gaussian.decode(r))


def test_inactivation_correct_bits(code: komm.abc.BlockCode):
    # Every returned bit matches the true message.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.5)
    decoder = komm.InactivationDecoder(code)
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        u_hat = decoder.decode(r)
        determined = u_hat != 2
        np.testing.assert_equal(u_hat[determined], u[determined])


def test_inactivation_no_erasures(code: komm.abc.BlockCode):
    # Without erasures, every message is recovered.
    decoder = komm.InactivationDecoder(code)
    k = code.dimension
    messages = komm.int_to_bits(range(2**k), width=k).reshape(-1, k)
    np.testing.assert_equal(decoder.decode(code.codewords()), messages)


def test_inactivation_sparse_check_matrix():
    # The intended use case, with plenty of inactivation.
    rng = np.random.default_rng(seed=42)
    m, n = 30, 60
    H = np.zeros((m, n), dtype=int)
    for j in range(n):
        H[rng.choice(m, 3, replace=False), j] = 1
    code = komm.BlockCode(check_matrix=H)
    dec_inactivation = komm.InactivationDecoder(code)
    dec_gaussian = komm.GaussianEliminationDecoder(code)
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.4)
    for _ in range(20):
        r = bec.transmit(code.encode(dms.emit(code.dimension)))
        np.testing.assert_equal(dec_inactivation.decode(r), dec_gaussian.decode(r))


@pytest.mark.parametrize(
    "r",
    [
        [1, 1, 0, 3, 0, 1, 1],
        [-1.3, -0.8, 1.1, -0.8, 1.2, -0.2, -1.4],
        [1.3, 0.8, 1.1, 0.8, 1.2, 0.2, 1.4],
    ],
)
def test_inactivation_invalid_input(r: list[float]):
    # Only bits and erasures are accepted.
    decoder = komm.InactivationDecoder(komm.HammingCode(3))
    with pytest.raises(ValueError):
        decoder.decode(r)
