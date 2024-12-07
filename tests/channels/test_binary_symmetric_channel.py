import numpy as np
import pytest

import komm
import komm.abc


def test_bsc_protocol():
    channel: komm.abc.DiscreteMemorylessChannel = komm.BinarySymmetricChannel(0.5)
    assert isinstance(channel, komm.abc.DiscreteMemorylessChannel)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_bsc_invalid_crossover_probability(p):
    with pytest.raises(ValueError):
        komm.BinarySymmetricChannel(p)


def test_bsc_noiseless():
    x = np.random.randint(2, size=10000)
    bsc0 = komm.BinarySymmetricChannel(0.0)
    assert np.array_equal(bsc0(x), x)
    bsc1 = komm.BinarySymmetricChannel(1.0)
    assert np.array_equal(bsc1(x), 1 - x)


@pytest.mark.parametrize(
    "p, pmf, base, expected",
    [
        (0.25, [0.5, 0.5], 2, 0.18872187554086717),
        (0.25, [0.5, 0.5], 4, 0.09436093777043358),
        (0.25, [0.1, 0.9], 2, 0.07001277477155976),
    ],
)
def test_bsc_mutual_information(p, pmf, base, expected):
    bsc = komm.BinarySymmetricChannel(p)
    assert np.isclose(bsc.mutual_information(pmf, base=base), expected)


@pytest.mark.parametrize(
    "p, base, expected",
    [
        (0.25, 2, 0.18872187554086717),
        (0.25, 4, 0.09436093777043358),
        (0.0, 2, 1.0),
        (0.5, 2, 0.0),
        (1.0, 2, 1.0),
    ],
)
def test_bsc_capacity(p, base, expected):
    bsc = komm.BinarySymmetricChannel(p)
    assert np.isclose(bsc.capacity(base=base), expected)


@pytest.mark.parametrize("p", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
@pytest.mark.parametrize("pi", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_bsc_mutual_information_dmc(p, base, pi):
    bsc1 = komm.BinarySymmetricChannel(p)
    bsc2 = komm.DiscreteMemorylessChannel([[1 - p, p], [p, 1 - p]])
    assert np.isclose(
        bsc1.mutual_information([1 - pi, pi], base=base),
        bsc2.mutual_information([1 - pi, pi], base=base),
    )


@pytest.mark.parametrize("p", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
def test_bsc_capacity_dmc(p, base):
    bsc1 = komm.BinarySymmetricChannel(p)
    bsc2 = komm.DiscreteMemorylessChannel([[1 - p, p], [p, 1 - p]])
    assert np.isclose(bsc1.capacity(base=base), bsc2.capacity(base=base))
