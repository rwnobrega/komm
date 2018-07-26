import pytest

import numpy as np
import komm


def test_scalar_quantizer():
    quantizer = komm.ScalarQuantizer(levels=[-1.0, 0.0, 1.2], thresholds=[-0.5, 0.8])
    assert np.allclose(quantizer([-1.01, -1.0, -0.99, -0.51, -0.5, -0.49]), [-1, -1, -1, -1, -1, 0])
    assert np.allclose(quantizer([0.8, -0.5]), [0.0, -1.0])
    assert np.allclose(quantizer([-1.0, 0.0, 1.2]), [-1.0, 0.0, 1.2])
    assert np.allclose(quantizer([-np.inf, np.inf]), [-1.0, 1.2])
