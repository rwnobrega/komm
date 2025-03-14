import numpy as np

from komm._modulation.labelings import (
    labeling_natural,
    labeling_natural_2d,
    labeling_reflected,
    labeling_reflected_2d,
)


def test_labeling_natural():
    assert np.array_equal(
        labeling_natural(2),
        [[0], [1]],
    )
    assert np.array_equal(
        labeling_natural(4),
        [[0, 0], [0, 1], [1, 0], [1, 1]],
    )
    assert np.array_equal(
        labeling_natural(8),
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


def test_labeling_reflected():
    assert np.array_equal(
        labeling_reflected(2),
        [[0], [1]],
    )
    assert np.array_equal(
        labeling_reflected(4),
        [[0, 0], [0, 1], [1, 1], [1, 0]],
    )
    # There is a typo in [AS15, Fig. 2.12], where the last two columns are swapped.
    assert np.array_equal(
        labeling_reflected(8),
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ],
    )


def test_labeling_reflected_2d():
    assert np.array_equal(
        labeling_reflected_2d((2, 2)),
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
    )
    assert np.array_equal(
        labeling_reflected_2d((2, 4)),
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
    )
    assert np.array_equal(
        labeling_reflected_2d((4, 2)),
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
    )
    assert np.array_equal(
        labeling_reflected_2d((4, 4)),
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
        ],
    )


def test_labeling_natural_2d():
    assert np.array_equal(
        labeling_natural_2d((2, 2)),
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
    )
    assert np.array_equal(
        labeling_natural_2d((2, 4)),
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
        ],
    )
    assert np.array_equal(
        labeling_natural_2d((4, 2)),
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    )
    assert np.array_equal(
        labeling_natural_2d((4, 4)),
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
    )
