import pytest

import numpy as np
import komm


def test_scalar_quantizer():
    quantizer = komm.ScalarQuantizer(levels=[-1.0, 0.0, 1.2], thresholds=[-0.5, 0.8])
    assert np.allclose(quantizer([-1.01, -1.0, -0.99, -0.51, -0.5, -0.49]), [-1, -1, -1, -1, 0, 0])
    assert np.allclose(quantizer([0.8, -0.5]), [1.2, 0.0])
    assert np.allclose(quantizer([-1.0, 0.0, 1.2]), [-1.0, 0.0, 1.2])
    assert np.allclose(quantizer([-np.inf, np.inf]), [-1.0, 1.2])


def test_uniform_quantizer_1():
    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice='unsigned')
    assert np.allclose(quantizer.levels, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    assert np.allclose(quantizer.thresholds, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25])

    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice='mid-riser')
    assert np.allclose(quantizer.levels, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    assert np.allclose(quantizer.thresholds, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice='mid-tread')
    assert np.allclose(quantizer.levels, [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert np.allclose(quantizer.thresholds, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


@pytest.mark.parametrize('choice', ['unsigned', 'mid-riser', 'mid-tread'])
@pytest.mark.parametrize('input_peak', [1.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize('num_levels', [2, 4, 8, 16, 32])
def test_uniform_quantizer_2(num_levels, input_peak, choice):
    quantizer_1 = komm.UniformQuantizer(num_levels, input_peak, choice)
    quantizer_2 = komm.ScalarQuantizer(quantizer_1.levels, quantizer_1.thresholds)
    x = np.linspace(-20.0, 20.0, num=10000)
    assert np.allclose(quantizer_1(x), quantizer_2(x))
