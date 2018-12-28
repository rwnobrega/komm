import numpy as np
import komm


def test_bsc():
    x = np.random.randint(2, size=10000)

    bsc0 = komm.BinarySymmetricChannel(0.0)
    assert np.array_equal(bsc0(x), x)

    bsc1 = komm.BinarySymmetricChannel(1.0)
    assert np.array_equal(bsc1(x), 1 - x)


def test_bec():
    x = np.random.randint(2, size=10000)

    bec0 = komm.BinaryErasureChannel(0.0)
    assert np.array_equal(bec0(x), x)

    bec1 = komm.BinaryErasureChannel(1.0)
    assert np.array_equal(bec1(x), np.full_like(x, fill_value=2))
