import numpy as np
import pytest

import komm


def test_discrete_memoryless_source_init():
    pmf = [0.5, 0.4, 0.1]
    source = komm.DiscreteMemorylessSource(pmf)
    assert np.array_equal(source.pmf, pmf)
    assert source.cardinality == 3


def test_discrete_memoryless_source_invalid():
    with pytest.raises(ValueError, match="pmf must be a 1D-array"):
        komm.DiscreteMemorylessSource([[0.25, 0.25], [0.25, 0.25]])
    with pytest.raises(ValueError, match="pmf must sum to 1.0"):
        komm.DiscreteMemorylessSource([0.5, 0.5, 0.1])
    with pytest.raises(ValueError, match="pmf must be non-negative"):
        komm.DiscreteMemorylessSource([0.5, -1.5])
    with pytest.raises(ValueError, match="cardinality must be at least 1"):
        komm.DiscreteMemorylessSource(0)


@pytest.mark.parametrize(
    "pmf",
    [
        [0.5, 0.4, 0.1],
        [0.3, 0.7],
    ],
)
def test_discrete_memoryless_source_output(pmf):
    source = komm.DiscreteMemorylessSource(pmf)
    symbols = source.emit((100, 500))
    assert symbols.shape == (100, 500)
    assert np.all(symbols >= 0) and np.all(symbols < source.cardinality)


@pytest.mark.parametrize(
    "pmf, x_entropy_base_2, x_entropy_base_3",
    [
        ([0.5, 0.4, 0.1], 1.360964047, 0.8586727111),
        ([0.3, 0.7], 0.8812908992, 0.5560326499),
    ],
)
def test_discrete_memoryless_source_entropy(pmf, x_entropy_base_2, x_entropy_base_3):
    source = komm.DiscreteMemorylessSource(pmf)
    assert np.allclose(source.entropy_rate(), x_entropy_base_2)
    assert np.allclose(source.entropy_rate(base=3), x_entropy_base_3)


@pytest.mark.parametrize(
    "pmf, x_symbol",
    [
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([0.0, 0.0, 1.0], 2),
    ],
)
def test_discrete_memoryless_source_constant(pmf, x_symbol):
    source = komm.DiscreteMemorylessSource(pmf)
    assert np.all(source.emit(1000) == x_symbol)
