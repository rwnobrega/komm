import numpy as np
import pytest

import komm


@pytest.mark.parametrize("pmf", [[0.5, 0.4, 0.1], [0.3, 0.7]])
def test_discrete_memoryless_source_output(pmf):
    dms = komm.DiscreteMemorylessSource(pmf)
    symbols = dms(1000)
    assert symbols.size == 1000
    assert np.all(symbols >= 0) and np.all(symbols < len(pmf))


@pytest.mark.parametrize(
    "pmf, x_entropy_base_2, x_entropy_base_3",
    [
        ([0.5, 0.4, 0.1], 1.360964047, 0.8586727111),
        ([0.3, 0.7], 0.8812908992, 0.5560326499),
    ],
)
def test_discrete_memoryless_source_entropy(pmf, x_entropy_base_2, x_entropy_base_3):
    dms = komm.DiscreteMemorylessSource(pmf)
    assert np.allclose(dms.entropy(), x_entropy_base_2)
    assert np.allclose(dms.entropy(base=3), x_entropy_base_3)


@pytest.mark.parametrize(
    "pmf, x_symbol",
    [
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([0.0, 0.0, 1.0], 2),
    ],
)
def test_discrete_memoryless_source_constant(pmf, x_symbol):
    dms = komm.DiscreteMemorylessSource(pmf)
    assert np.all(dms(1000) == x_symbol)
