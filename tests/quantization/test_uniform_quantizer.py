import numpy as np
import pytest

import komm
from komm._quantization.util import mean_squared_quantization_error


def test_uniform_quantizer_choices():
    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice="unsigned")
    assert np.allclose(quantizer.levels, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    assert np.allclose(quantizer.thresholds, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25])

    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice="mid-riser")
    assert np.allclose(quantizer.levels, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    assert np.allclose(quantizer.thresholds, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    quantizer = komm.UniformQuantizer(num_levels=8, input_peak=4.0, choice="mid-tread")
    assert np.allclose(quantizer.levels, [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert np.allclose(quantizer.thresholds, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


@pytest.mark.parametrize("choice", ["unsigned", "mid-riser", "mid-tread"])
@pytest.mark.parametrize("input_peak", [1.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize("num_levels", [2, 4, 8, 16, 32])
def test_uniform_quantizer_scalar_equivalent(num_levels, input_peak, choice):
    quantizer_1 = komm.UniformQuantizer(num_levels, input_peak, choice)
    quantizer_2 = komm.ScalarQuantizer(quantizer_1.levels, quantizer_1.thresholds)
    x = np.linspace(-20.0, 20.0, num=10000)
    assert np.allclose(quantizer_1(x), quantizer_2(x))


def test_uniform_quantizer_invalid_constructions():
    with pytest.raises(ValueError):
        komm.UniformQuantizer(num_levels=1)
    with pytest.raises(ValueError):
        komm.UniformQuantizer(num_levels=8, choice="invalid")  # type: ignore


@pytest.mark.parametrize("n_bits", [2, 3, 4, 5])
@pytest.mark.parametrize("input_peak", [0.5, 1.0, 1.5, 2.0])
def test_uniform_quantizer_snr(n_bits, input_peak):
    quantizer = komm.UniformQuantizer(num_levels=2**n_bits, input_peak=input_peak)
    signal_power = (2 * input_peak) ** 2 / 12
    noise_power = mean_squared_quantization_error(
        quantizer,
        input_pdf=lambda x: 1 / (2 * input_peak) * (np.abs(x) <= input_peak),
        input_range=(-input_peak, input_peak),
        points_per_interval=1000,
    )
    snr_db = 10 * np.log10(signal_power / noise_power)
    assert np.isclose(snr_db, 6.02 * n_bits, atol=0.05)
