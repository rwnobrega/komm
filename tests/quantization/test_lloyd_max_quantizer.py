from functools import partial

import numpy as np
import pytest

import komm
from komm._quantization.util import mean_squared_quantization_error

uniform_pdf = lambda x, peak: 1 / (2 * peak) * (np.abs(x) <= peak)
gaussian_pdf = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)
laplacian_pdf = lambda x: 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.abs(x))


@pytest.mark.parametrize("num_levels", range(2, 21))
@pytest.mark.parametrize("peak", [1, 2, 3, 4, 5])
def test_lloyd_max_quantizer_uniform(num_levels, peak):
    quantizer1 = komm.LloydMaxQuantizer(
        input_pdf=partial(uniform_pdf, peak=peak),
        input_range=(-peak, peak),
        num_levels=num_levels,
    )
    quantizer2 = komm.UniformQuantizer(
        num_levels=num_levels,
        input_peak=peak,
        choice="mid-riser" if num_levels % 2 == 0 else "mid-tread",
    )
    assert np.allclose(quantizer1.levels, quantizer2.levels, atol=1e-5)
    assert np.allclose(quantizer1.thresholds, quantizer2.thresholds, atol=1e-5)


test_cases = [
    {
        "input_pdf": gaussian_pdf,
        "input_range": (-5, 5),
        "num_levels": 4,
        "levels": [-1.510, -0.4528, 0.4528, 1.510],
        "thresholds": [-0.9816, 0.0, 0.9816],
        "snr_db": 9.3,
    },
    {
        "input_pdf": gaussian_pdf,
        "input_range": (-5, 5),
        "num_levels": 6,
        "levels": [-1.894, -1.0, -0.3177, 0.3177, 1.0, 1.894],
        "thresholds": [-1.447, -0.6589, 0.0, 0.6589, 1.447],
        "snr_db": 12.41,
    },
    {
        # In the book, level "0.7560" is "0.6812" and threshold "0.5006" is "0.7560".
        "input_pdf": gaussian_pdf,
        "input_range": (-5, 5),
        "num_levels": 8,
        "levels": [-2.152, -1.344, -0.7560, -0.2451, 0.2451, 0.7560, 1.344, 2.152],
        "thresholds": [-1.748, -1.050, -0.5006, 0.0, 0.5006, 1.050, 1.748],
        "snr_db": 14.62,
    },
    {
        "input_pdf": laplacian_pdf,
        "input_range": (-10, 10),
        "num_levels": 4,
        "levels": [-1.8340, -0.4196, 0.4196, 1.8340],
        "thresholds": [-1.1269, 0.0, 1.1269],
        "snr_db": 7.54,
    },
    {
        "input_pdf": laplacian_pdf,
        "input_range": (-10, 10),
        "num_levels": 6,
        "levels": [-2.5535, -1.1393, -0.2998, 0.2998, 1.1393, 2.5535],
        "thresholds": [-1.8464, -0.7195, 0.0, 0.7195, 1.8464],
        "snr_db": 10.51,
    },
    {
        "input_pdf": laplacian_pdf,
        "input_range": (-10, 10),
        "num_levels": 8,
        "levels": [-3.0867, -1.6725, -0.8330, -0.2334, 0.2334, 0.8330, 1.6725, 3.0867],
        "thresholds": [-2.3796, -1.2527, -0.5332, 0.0, 0.5332, 1.2527, 2.3796],
        "snr_db": 12.64,
    },
]


@pytest.mark.parametrize(
    "input_pdf, input_range, num_levels, levels, thresholds, snr_db",
    [
        (
            case["input_pdf"],
            case["input_range"],
            case["num_levels"],
            case["levels"],
            case["thresholds"],
            case["snr_db"],
        )
        for case in test_cases
    ],
)
def test_lloyd_max_quantizer_sayood_table_9_6(
    input_pdf, input_range, num_levels, levels, thresholds, snr_db
):
    # [Say06, Sec. 9.6.1, Table.9.6]
    quantizer = komm.LloydMaxQuantizer(input_pdf, input_range, num_levels)
    assert np.allclose(quantizer.levels, levels, atol=0.0005)
    assert np.allclose(quantizer.thresholds, thresholds, atol=0.0005)
    x = np.linspace(input_range[0], input_range[1], num=1000)
    signal_power = np.trapezoid(input_pdf(x) * x**2, x)
    noise_power = mean_squared_quantization_error(
        quantizer, input_pdf, input_range, points_per_interval=1000
    )
    assert np.isclose(10 * np.log10(signal_power / noise_power), snr_db, atol=0.05)
