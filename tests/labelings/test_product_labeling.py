import numpy as np
import pytest

import komm

reflected_1 = komm.ReflectedLabeling(1)
reflected_2 = komm.ReflectedLabeling(2)


def test_labeling_reflected_pair():
    assert np.array_equal(
        komm.ProductLabeling(reflected_1, reflected_1).matrix,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
    )
    assert np.array_equal(
        komm.ProductLabeling(reflected_1, reflected_2).matrix,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
    )
    assert np.array_equal(
        komm.ProductLabeling(reflected_2, reflected_1).matrix,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
    )
    assert np.array_equal(
        komm.ProductLabeling(reflected_2, reflected_2).matrix,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
        ],
    )


def test_labeling_reflected_triple():
    assert np.array_equal(
        komm.ProductLabeling(reflected_1, reflected_1, reflected_1).matrix,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
    )


def test_labeling_reflected_repeat():
    assert np.array_equal(
        komm.ProductLabeling(reflected_1, repeat=2).matrix,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
    )
    assert np.array_equal(
        komm.ProductLabeling(reflected_1, repeat=3).matrix,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
    )
    assert np.array_equal(
        komm.ProductLabeling(reflected_2, repeat=2).matrix,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
        ],
    )


def test_labeling_product_invalid_repeat():
    labeling = komm.ReflectedLabeling(1)
    with pytest.raises(ValueError, match=r"'repeat' must be at least 1 \(got 0\)"):
        komm.ProductLabeling(labeling, repeat=0)
    with pytest.raises(TypeError, match="'repeat' must be an integer"):
        komm.ProductLabeling(labeling, repeat=2.0)  # type: ignore
