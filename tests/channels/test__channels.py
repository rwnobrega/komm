import numpy as np
import pytest

import komm
import komm.abc


def test_gaussian_vectorized_input(rng):
    channel = komm.GaussianChannel()  # noiseless
    x = rng.standard_normal((3, 4, 5))
    np.testing.assert_equal(x, channel.transmit(x))


@pytest.mark.parametrize(
    "channel",
    [
        komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2]]),
        komm.BinarySymmetricChannel(0.15),
        komm.BinaryErasureChannel(0.33),
        komm.ZChannel(0.42),
    ],
)
def test_discrete_channels_vectorized_input(
    channel: komm.abc.DiscreteMemorylessChannel, rng
):
    x = rng.integers(0, channel.input_cardinality, size=(3, 4, 5))
    y = channel.transmit(x)
    np.testing.assert_equal(x.shape, y.shape)


@pytest.mark.parametrize(
    "channel",
    [
        komm.GaussianChannel(noise_power=1.0),
        komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2]]),
        komm.BinarySymmetricChannel(0.15),
        komm.BinaryErasureChannel(0.33),
        komm.ZChannel(0.42),
    ],
)
def test_channels_follow_global_rng(channel):
    x = np.ones(100, dtype=int)
    komm.global_rng.set(np.random.default_rng(1))
    y = channel.transmit(x)
    komm.global_rng.set(np.random.default_rng(1))
    np.testing.assert_equal(channel.transmit(x), y)


def test_channels_equality_ignores_rng():
    channel = komm.BinarySymmetricChannel(0.1, rng=np.random.default_rng(1))
    assert channel == komm.BinarySymmetricChannel(0.1)
