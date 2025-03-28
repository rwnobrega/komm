import numpy as np

import komm


def test_scalar_quantizer():
    quantizer = komm.ScalarQuantizer(levels=[-1.0, 0.0, 1.2], thresholds=[-0.5, 0.8])
    input = [0.79, 0.8, 0.81, -1.01, -1.0, -0.99, -0.51, -0.5, -0.49]
    assert np.allclose(quantizer.digitize(input), [1, 2, 2, 0, 0, 0, 0, 1, 1])
    assert np.allclose(quantizer.quantize(input), [0, 1.2, 1.2, -1, -1, -1, -1, 0, 0])
    assert np.allclose(quantizer.quantize([0.8, -0.5]), [1.2, 0.0])
    assert np.allclose(quantizer.quantize([-1.0, 0.0, 1.2]), [-1.0, 0.0, 1.2])
    assert np.allclose(quantizer.quantize([-np.inf, np.inf]), [-1.0, 1.2])
