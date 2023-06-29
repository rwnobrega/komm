import numpy as np

import komm


def test_labeling_natural():
    assert np.array_equal(
        komm.Modulation._labeling_natural(2),
        [[0], [1]],
    )
    assert np.array_equal(
        komm.Modulation._labeling_natural(4),
        [[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    assert np.array_equal(
        komm.Modulation._labeling_natural(8),
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    )


def test_labeling_reflected():
    assert np.array_equal(
        komm.Modulation._labeling_reflected(2),
        [[0], [1]],
    )
    assert np.array_equal(
        komm.Modulation._labeling_reflected(4),
        [[0, 0], [1, 0], [1, 1], [0, 1]],
    )
    # There is a typo in [AS15, Fig. 2.12], where the last two columns are swapped.
    assert np.array_equal(
        komm.Modulation._labeling_reflected(8),
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]],
    )


def test_labeling_reflected_2d():
    assert np.array_equal(
        komm.Modulation._labeling_reflected_2d(2, 2),
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
    )
    assert np.array_equal(
        komm.Modulation._labeling_reflected_2d(2, 4),
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
    )
    assert np.array_equal(
        komm.Modulation._labeling_reflected_2d(4, 2),
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
    )
    assert np.array_equal(
        komm.Modulation._labeling_reflected_2d(4, 4),
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 0, 1],
        ],
    )
