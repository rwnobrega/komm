from functools import partial

import numpy as np
import pytest

import komm
from komm._quantization.util import mean_squared_quantization_error


@pytest.mark.parametrize(
    "num_levels, input_range, choice, levels, thresholds",
    [
        (
            8,
            (-4.0, 4.0),
            "mid-riser",
            [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        ),
        (
            8,
            (-4.0, 4.0),
            "mid-tread",
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        ),
        (
            8,
            (0.0, 4.0),
            "mid-riser",
            [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        ),
        (
            8,
            (0.0, 4.0),
            "mid-tread",
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
        ),
    ],
)
def test_uniform_quantizer_choices(num_levels, input_range, choice, levels, thresholds):
    quantizer = komm.UniformQuantizer(
        num_levels=num_levels, input_range=input_range, choice=choice
    )
    assert np.allclose(quantizer.levels, levels)
    assert np.allclose(quantizer.thresholds, thresholds)


def test_uniform_quantizer_range():
    quantizer = komm.UniformQuantizer(num_levels=8)
    x = np.linspace(-0.5, 0.4, num=10)
    y = [-0.375, -0.375, -0.375, -0.125, -0.125, 0.125, 0.125, 0.125, 0.375, 0.375]
    assert np.allclose(quantizer.quantize(x), y)


@pytest.mark.parametrize("choice", ["mid-riser", "mid-tread"])
@pytest.mark.parametrize("peak", [1.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize("num_levels", [2, 4, 8, 16, 32])
def test_uniform_quantizer_scalar_equivalent(num_levels, peak, choice):
    input_range = (-peak, peak)
    quantizer_1 = komm.UniformQuantizer(num_levels, input_range, choice)
    quantizer_2 = komm.ScalarQuantizer(quantizer_1.levels, quantizer_1.thresholds)
    x = np.linspace(-20.0, 20.0, num=10000)
    assert np.allclose(quantizer_1.quantize(x), quantizer_2.quantize(x))


def test_uniform_quantizer_invalid_constructions():
    with pytest.raises(ValueError):
        komm.UniformQuantizer(num_levels=1)
    with pytest.raises(ValueError):
        komm.UniformQuantizer(num_levels=8, choice="invalid")  # type: ignore


@pytest.mark.parametrize("n_bits", [2, 3, 4, 5])
@pytest.mark.parametrize("peak", [0.5, 1.0, 1.5, 2.0])
def test_uniform_quantizer_snr(n_bits, peak):
    uniform_pdf = lambda x, peak: 1 / (2 * peak) * (np.abs(x) <= peak)

    input_range = (-peak, peak)
    quantizer = komm.UniformQuantizer(num_levels=2**n_bits, input_range=input_range)
    signal_power = (2 * peak) ** 2 / 12
    noise_power = mean_squared_quantization_error(
        quantizer,
        input_pdf=partial(uniform_pdf, peak=peak),
        input_range=input_range,
        points_per_interval=1000,
    )
    snr_db = 10 * np.log10(signal_power / noise_power)
    assert np.isclose(snr_db, 6.02 * n_bits, atol=0.05)
