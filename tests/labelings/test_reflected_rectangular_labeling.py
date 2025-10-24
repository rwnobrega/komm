import numpy as np
import pytest

import komm


def test_labeling_reflected_retangular_tuple():
    assert np.array_equal(
        komm.ReflectedRectangularLabeling((1, 1)).matrix,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
    )
    assert np.array_equal(
        komm.ReflectedRectangularLabeling((1, 2)).matrix,
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
        komm.ReflectedRectangularLabeling((2, 1)).matrix,
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
        komm.ReflectedRectangularLabeling((2, 2)).matrix,
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


def test_labeling_reflected_retangular_int():
    assert np.array_equal(
        komm.ReflectedRectangularLabeling(2).matrix,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
    )
    assert np.array_equal(
        komm.ReflectedRectangularLabeling(4).matrix,
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


def test_labeling_reflected_retangular_invalid():
    with pytest.raises(ValueError, match="must contain positive integers"):
        komm.ReflectedRectangularLabeling((-1, 2))
    with pytest.raises(ValueError, match="must be an even number"):
        komm.ReflectedRectangularLabeling(-4)
    with pytest.raises(ValueError, match="must be an even number"):
        komm.ReflectedRectangularLabeling(5)
