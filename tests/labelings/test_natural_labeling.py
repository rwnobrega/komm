import numpy as np

import komm


def test_labeling_natural():
    lab = komm.NaturalLabeling(1)
    assert np.array_equal(
        lab.matrix,
        [
            [0],
            [1],
        ],
    )
    lab = komm.NaturalLabeling(2)
    assert np.array_equal(
        lab.matrix,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
    )
    lab = komm.NaturalLabeling(3)
    assert np.array_equal(
        lab.matrix,
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
