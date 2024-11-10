import numpy as np
import pytest

import komm
from komm._channels.AbstractDiscreteMemorylessChannel import (
    AbstractDiscreteMemorylessChannel,
)


def test_bec_protocol():
    channel: AbstractDiscreteMemorylessChannel = komm.BinaryErasureChannel(0.5)
    assert isinstance(channel, AbstractDiscreteMemorylessChannel)


@pytest.mark.parametrize("eps", [-0.1, 1.1])
def test_bec_invalid_erasure_probability(eps):
    with pytest.raises(ValueError):
        komm.BinaryErasureChannel(eps)


def test_bec_noiseless():
    x = np.random.randint(2, size=10000)
    bec = komm.BinaryErasureChannel(0.0)
    assert np.array_equal(bec(x), x)


def test_bec_useless():
    x = np.random.randint(2, size=10000)
    bec = komm.BinaryErasureChannel(1.0)
    assert np.array_equal(bec(x), np.full_like(x, fill_value=2))


@pytest.mark.parametrize(
    "eps, pmf, base, expected",
    [
        (0.25, [0.5, 0.5], 2, 0.75),
        (0.25, [0.5, 0.5], 4, 0.375),
        (0.25, [0.1, 0.9], 2, 0.3517466951919609),
    ],
)
def test_bec_mutual_information(eps, pmf, base, expected):
    bec = komm.BinaryErasureChannel(eps)
    assert np.isclose(bec.mutual_information(pmf, base=base), expected)


@pytest.mark.parametrize(
    "eps, base, expected",
    [
        (0.25, 2, 0.75),
        (0.25, 4, 0.375),
        (0.0, 2, 1.0),
        (1.0, 2, 0.0),
    ],
)
def test_bec_capacity(eps, base, expected):
    bec = komm.BinaryErasureChannel(eps)
    assert np.isclose(bec.capacity(base=base), expected)


@pytest.mark.parametrize("eps", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
@pytest.mark.parametrize("pi", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_bec_mutual_information_dmc(eps, base, pi):
    bec1 = komm.BinaryErasureChannel(eps)
    bec2 = komm.DiscreteMemorylessChannel([[1 - eps, 0, eps], [0, 1 - eps, eps]])
    assert np.isclose(
        bec1.mutual_information([1 - pi, pi], base=base),
        bec2.mutual_information([1 - pi, pi], base=base),
    )


@pytest.mark.parametrize("eps", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("base", [2.0, 3.0, 4.0, 10.0])
def test_bec_capacity_dmc(eps, base):
    bec1 = komm.BinaryErasureChannel(eps)
    bec2 = komm.DiscreteMemorylessChannel([[1 - eps, 0, eps], [0, 1 - eps, eps]])
    assert np.isclose(bec1.capacity(base=base), bec2.capacity(base=base))
