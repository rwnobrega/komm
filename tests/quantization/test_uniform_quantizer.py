from functools import partial

import numpy as np
import pytest

import komm

from .util import gaussian_pdf, laplacian_pdf, uniform_pdf


def test_uniform_quantizer_mid_riser():
    quantizer = komm.UniformQuantizer.mid_riser(num_levels=8, step=1.0)
    assert np.allclose(quantizer.levels, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    assert np.allclose(quantizer.thresholds, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def test_uniform_quantizer_mid_tread():
    quantizer = komm.UniformQuantizer.mid_tread(num_levels=8, step=1.0)
    assert np.allclose(quantizer.levels, [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert np.allclose(quantizer.thresholds, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


def test_uniform_quantizer_positive():
    quantizer = komm.UniformQuantizer(num_levels=8, step=0.5, offset=4.0)
    assert np.allclose(
        quantizer.levels,
        [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
    )
    assert np.allclose(
        quantizer.thresholds,
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    )

    quantizer = komm.UniformQuantizer(num_levels=8, step=0.5, offset=3.5)
    assert np.allclose(
        quantizer.levels,
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    )
    assert np.allclose(
        quantizer.thresholds,
        [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
    )


def test_uniform_quantizer_range():
    quantizer = komm.UniformQuantizer(num_levels=8, step=1 / 4)
    x = np.linspace(-0.5, 0.4, num=10)
    y = [-0.375, -0.375, -0.375, -0.125, -0.125, 0.125, 0.125, 0.125, 0.375, 0.375]
    assert np.allclose(quantizer.quantize(x), y)


@pytest.mark.parametrize("peak", [1.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize("num_levels", [2, 4, 8, 16, 32])
def test_uniform_quantizer_mid_riser_scalar_equivalent(num_levels, peak):
    quantizer_1 = komm.UniformQuantizer.mid_riser(num_levels, peak)
    quantizer_2 = komm.ScalarQuantizer(quantizer_1.levels, quantizer_1.thresholds)
    x = np.linspace(-20.0, 20.0, num=10000)
    assert np.allclose(quantizer_1.quantize(x), quantizer_2.quantize(x))


@pytest.mark.parametrize("peak", [1.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
@pytest.mark.parametrize("num_levels", [2, 4, 8, 16, 32])
def test_uniform_quantizer_mid_tread_scalar_equivalent(num_levels, peak):
    quantizer_1 = komm.UniformQuantizer.mid_tread(num_levels, peak)
    quantizer_2 = komm.ScalarQuantizer(quantizer_1.levels, quantizer_1.thresholds)
    x = np.linspace(-20.0, 20.0, num=10000)
    assert np.allclose(quantizer_1.quantize(x), quantizer_2.quantize(x))


def test_uniform_quantizer_invalid_constructions():
    with pytest.raises(ValueError, match="must be greater than 1"):
        komm.UniformQuantizer(num_levels=1, step=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        komm.UniformQuantizer(num_levels=8, step=0.0)


@pytest.mark.parametrize("n_bits", [2, 3, 4, 5])
@pytest.mark.parametrize("peak", [0.5, 1.0, 1.5, 2.0])
def test_uniform_quantizer_snr(n_bits, peak):
    num_levels = 2**n_bits
    step = 2 * peak / num_levels
    quantizer = komm.UniformQuantizer.mid_riser(num_levels, step)
    noise_power = quantizer.mean_squared_error(
        input_pdf=partial(uniform_pdf, peak=peak),
        input_range=(-peak, peak),
    )
    assert np.isclose(noise_power, quantizer.step**2 / 12)
    signal_power = (2 * peak) ** 2 / 12
    snr_db = 10 * np.log10(signal_power / noise_power)
    assert np.isclose(snr_db, 6.02059991328 * n_bits)


@pytest.mark.parametrize(
    "num_levels, step, mse",
    [
        (2, 1.596, 0.3634),
        (3, 1.224, 0.1902),
        (4, 0.9957, 0.1188),
        (5, 0.8430, 0.08218),
        (6, 0.7334, 0.06065),
        (7, 0.6508, 0.04686),
        (8, 0.5860, 0.03744),
        (9, 0.5338, 0.03069),
        (10, 0.4908, 0.02568),
        (11, 0.4546, 0.02185),
        (12, 0.4238, 0.01885),
        (13, 0.3972, 0.01645),
        (14, 0.3739, 0.01450),
        (15, 0.3534, 0.01289),
        (16, 0.3352, 0.01154),
        (17, 0.3189, 0.01040),
        (18, 0.3042, 0.009430),
        (19, 0.2909, 0.008594),
        (20, 0.2788, 0.007869),
        (21, 0.2678, 0.007235),
        (22, 0.2576, 0.006678),
        (23, 0.2482, 0.006185),
        (24, 0.2396, 0.005747),
        (25, 0.2315, 0.005355),
        (26, 0.2240, 0.005004),
        (27, 0.2171, 0.004687),
        (28, 0.2105, 0.004401),
        (29, 0.2044, 0.004141),
        (30, 0.1987, 0.003905),
        (31, 0.1932, 0.003688),
        (32, 0.1881, 0.003490),
        (33, 0.1833, 0.003308),
        (34, 0.1787, 0.003141),
        (35, 0.1744, 0.002986),
        (36, 0.1703, 0.002843),
    ],
)
def test_uniform_quantizer_gaussian(num_levels, step, mse):
    # Joel Max (1960):
    # Quantizing for Minimum Distortion.
    # Table II.
    quantizer = komm.UniformQuantizer(num_levels=num_levels, step=step)
    assert np.isclose(
        quantizer.mean_squared_error(input_pdf=gaussian_pdf, input_range=(-6, 6)),
        mse,
        atol=5e-5,
    )


@pytest.mark.parametrize(
    "num_levels, step, mse",
    [
        (2, 1.4142, 0.5000),
        (3, 1.4142, 0.2642),
        (4, 1.0874, 0.1936),
        (5, 1.0245, 0.1330),
        (6, 0.8707, 0.1095),
        (7, 0.8218, 0.0831),
        (8, 0.7309, 0.0717),
        (9, 0.6490, 0.0580),
        (10, 0.6335, 0.0515),
        (11, 0.6048, 0.0433),
        (12, 0.5613, 0.0392),
        (13, 0.5385, 0.0339),
        (14, 0.5056, 0.0310),
        (15, 0.4869, 0.0274),
        (16, 0.4610, 0.0254),
        (17, 0.4454, 0.0227),
        (18, 0.4245, 0.0210),
        (19, 0.4113, 0.0192),
        (20, 0.3940, 0.0180),
        (21, 0.3826, 0.0165),
        (22, 0.3680, 0.0156),
        (23, 0.3581, 0.0144),
        (24, 0.3456, 0.0136),
        (25, 0.3369, 0.0127),
        (26, 0.3261, 0.0120),
        (27, 0.3184, 0.0112),
        (28, 0.3089, 0.0107),
        (29, 0.3020, 0.0101),
        (30, 0.2937, 0.0096),
        (31, 0.2875, 0.0091),
        (32, 0.2800, 0.0087),
    ],
)
def test_uniform_quantizer_laplacian(num_levels, step, mse):
    # W. C. Adams, Jr. & C. E. Giesler (1978):
    # Quantizing Characteristics for Signals Having Laplacian Amplitude Probability Density Function.
    # Table 1.
    quantizer = komm.UniformQuantizer(num_levels=num_levels, step=step)
    assert np.isclose(
        quantizer.mean_squared_error(input_pdf=laplacian_pdf, input_range=(-10, 10)),
        mse,
        atol=5e-3,
    )
