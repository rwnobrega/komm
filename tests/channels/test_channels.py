import numpy as np
import pytest

import komm
import komm.abc


def test_awgn_vectorized_input():
    awgn = komm.AWGNChannel(signal_power=5.0, snr=np.inf)  # noiseless
    x = np.random.randn(3, 4, 5)
    np.testing.assert_array_equal(x, awgn(x))


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
    channel: komm.abc.DiscreteMemorylessChannel,
):
    x = np.random.randint(0, channel.input_cardinality, size=(3, 4, 5))
    y = channel(x)
    np.testing.assert_equal(x.shape, y.shape)
