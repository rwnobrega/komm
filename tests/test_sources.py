import numpy as np
import komm


def test_dms():
    dms0 = komm.DiscreteMemorylessSource([1.0, 0.0])
    assert np.array_equal(dms0(10000), np.zeros(10000, dtype=int))

    dms1 = komm.DiscreteMemorylessSource([0.0, 1.0])
    assert np.array_equal(dms1(10000), np.ones(10000, dtype=int))
