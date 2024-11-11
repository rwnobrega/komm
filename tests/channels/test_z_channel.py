import numpy as np
import pytest

import komm
from komm._channels.AbstractDiscreteMemorylessChannel import (
    AbstractDiscreteMemorylessChannel,
)


def test_zc_protocol():
    channel: komm.ZChannel = komm.ZChannel(0.5)
    assert isinstance(channel, AbstractDiscreteMemorylessChannel)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_zc_invalid_crossover_probability(p):
    with pytest.raises(ValueError):
        komm.ZChannel(p)


def test_zc_noiseless():
    x = np.random.randint(2, size=10000)
    zc = komm.ZChannel(0.0)
    assert np.array_equal(zc(x), x)


def test_zc_useless():
    x = np.random.randint(2, size=10000)
    zc = komm.ZChannel(1.0)
    assert np.array_equal(zc(x), np.zeros_like(x))


@pytest.mark.parametrize(
    "p, pmf, base, expected",
    [
        (0.5, [0.6, 0.4], 2, 0.3219280948873623),
        (0.5, [0.6, 0.4], 4, 0.16096404744368115),
        (0.5, [0.5, 0.5], 2, 0.31127812445913283),
    ],
)
def test_zc_mutual_information(p, pmf, base, expected):
    zc = komm.ZChannel(p)
    assert np.isclose(zc.mutual_information(pmf, base=base), expected)


@pytest.mark.parametrize(
    "p, base, expected",
    [
        (0.5, 2, 0.3219280948873623),
        (0.5, 4, 0.16096404744368115),
        (0.0, 2, 1.0),
        (0.25, 2, 0.5582386267373455),
        (1.0, 2, 0.0),
    ],
)
def test_zc_capacity(p, base, expected):
    zc = komm.ZChannel(p)
    assert np.isclose(zc.capacity(base=base), expected)


@pytest.mark.parametrize("p", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
@pytest.mark.parametrize("pi", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_zc_mutual_information_dmc(p, base, pi):
    zc1 = komm.ZChannel(p)
    zc2 = komm.DiscreteMemorylessChannel([[1, 0], [p, 1 - p]])
    assert np.isclose(
        zc1.mutual_information([1 - pi, pi], base=base),
        zc2.mutual_information([1 - pi, pi], base=base),
    )


@pytest.mark.parametrize("p", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
def test_zc_capacity_dmc(p, base):
    zc1 = komm.ZChannel(p)
    zc2 = komm.DiscreteMemorylessChannel([[1, 0], [p, 1 - p]])
    assert np.isclose(zc1.capacity(base=base), zc2.capacity(base=base))
