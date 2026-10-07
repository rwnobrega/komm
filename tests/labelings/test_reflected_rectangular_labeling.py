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
    with pytest.raises(ValueError, match="elements of 'num_bits' must be at least 1"):
        komm.ReflectedRectangularLabeling((-1, 2))
    with pytest.raises(TypeError, match="'num_bits' must contain only integers"):
        komm.ReflectedRectangularLabeling((2.0, 2))  # type: ignore
    with pytest.raises(ValueError, match=r"'num_bits' must be at least 2 \(got -4\)"):
        komm.ReflectedRectangularLabeling(-4)
    with pytest.raises(ValueError, match="must be an even number"):
        komm.ReflectedRectangularLabeling(5)
