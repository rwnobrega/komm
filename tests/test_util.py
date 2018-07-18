import pytest

import numpy as np
import komm


def test_entropy():
    assert np.allclose(komm.entropy([0.5, 0.5]), 1.0)
    assert np.allclose(komm.entropy([1.0, 0.0]), 0.0)
    assert np.allclose(komm.entropy([0.25, 0.75]), 2.0 - 0.75 * (np.log2(3.0)))
    assert np.allclose(komm.entropy([1/3, 1/3, 1/3]), np.log2(3))
    assert np.allclose(komm.entropy([1/3, 1/3, 1/3], base=3.0), 1.0)
    assert np.allclose(komm.entropy([1/3, 1/3, 1/3], base='e'), np.log(3.0))

def test_mutual_information():
    pass
