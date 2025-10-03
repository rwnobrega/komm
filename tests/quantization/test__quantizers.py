import numpy as np
import pytest

import komm
import komm.abc


@pytest.mark.parametrize(
    "quantizer",
    [
        komm.ScalarQuantizer(
            levels=[-7, -5, -3, -1, 1, 3, 5, 7],
            thresholds=[-6, -4, -2, 0, 2, 4, 6],
        ),
        komm.UniformQuantizer.mid_riser(num_levels=8, step=2.0),
    ],
)
def test_quantizers_on_threshold(quantizer: komm.abc.ScalarQuantizer):
    x = [-1e-10, 0.0, 1e-10]
    np.testing.assert_array_equal(quantizer.digitize(x), [3, 4, 4])
    np.testing.assert_array_equal(quantizer.quantize(x), [-1.0, 1.0, 1.0])
