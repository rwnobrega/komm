import numpy as np

import komm


def test_labeling_reflected():
    lab = komm.ReflectedLabeling(1)
    assert np.array_equal(
        lab.matrix,
        [
            [0],
            [1],
        ],
    )
    lab = komm.ReflectedLabeling(2)
    assert np.array_equal(
        lab.matrix,
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ],
    )
    lab = komm.ReflectedLabeling(3)
    # There is a typo in [AS15, Fig. 2.12], where the last two columns are swapped.
    assert np.array_equal(
        lab.matrix,
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
