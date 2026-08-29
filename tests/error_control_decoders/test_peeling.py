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


def test_peeling_exact_when_successful(code: komm.abc.BlockCode):
    # Whenever it succeeds, the message is recovered.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.4)
    decoder = komm.PeelingDecoder(code)
    successes = 0
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        u_hat = decoder.decode(r)
        if not np.any(u_hat == 2):
            np.testing.assert_equal(u_hat, u)
            successes += 1
    assert successes > 0


def test_peeling_no_erasures(code: komm.abc.BlockCode):
    # Without erasures, every message is recovered.
    decoder = komm.PeelingDecoder(code)
    k = code.dimension
    messages = komm.int_to_bits(range(2**k), width=k)
    np.testing.assert_equal(decoder.decode(code.codewords()), messages)


def test_peeling_all_erased(code: komm.abc.BlockCode):
    # Nothing to propagate from, so decoding fails.
    decoder = komm.PeelingDecoder(code)
    r = np.full(code.length, 2)
    np.testing.assert_equal(decoder.decode(r), np.full(code.dimension, 2))


def test_peeling_stopping_set():
    # Peeling stalls where the optimal decoder succeeds.
    code = komm.HammingCode(3)
    dec_peeling = komm.PeelingDecoder(code)
    dec_gaussian = komm.GaussianEliminationDecoder(code)
    r = [2, 2, 0, 2, 0, 1, 1]
    np.testing.assert_equal(dec_peeling.decode(r), [2, 2, 0, 2])
    np.testing.assert_equal(dec_gaussian.decode(r), [1, 1, 0, 0])


def test_peeling_correct_bits(code: komm.abc.BlockCode):
    # Every returned bit matches the true message.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.5)
    decoder = komm.PeelingDecoder(code)
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        u_hat = decoder.decode(r)
        determined = u_hat != 2
        np.testing.assert_equal(u_hat[determined], u[determined])


@pytest.mark.parametrize(
    "r",
    [
        [1, 1, 0, 3, 0, 1, 1],
        [-1.3, -0.8, 1.1, -0.8, 1.2, -0.2, -1.4],
        [1.3, 0.8, 1.1, 0.8, 1.2, 0.2, 1.4],
    ],
)
def test_peeling_invalid_input(r: list[float]):
    # Only bits and erasures are accepted.
    decoder = komm.PeelingDecoder(komm.HammingCode(3))
    with pytest.raises(ValueError):
        decoder.decode(r)
