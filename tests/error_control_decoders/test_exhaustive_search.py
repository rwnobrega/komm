import numpy as np

import komm


def test_exhaustive_search_hard():
    code = komm.HammingCode(3, extended=True)
    decoder = komm.ExhaustiveSearchDecoder(code, input_type="hard")
    np.testing.assert_equal(
        decoder.decode([
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 0],
        ]),
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ],
    )

    code = komm.SimplexCode(3)
    decoder = komm.ExhaustiveSearchDecoder(code, input_type="hard")
    np.testing.assert_equal(
        decoder.decode([
            [1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
        [
            [1, 0, 1],
            [0, 0, 0],
        ],
    )


def test_exhaustive_search_soft():
    code = komm.HammingCode(3)
    decoder = komm.ExhaustiveSearchDecoder(code, input_type="soft")
    np.testing.assert_equal(
        decoder.decode([
            [-0.98, -0.85, 1.07, -0.78, 1.11, -0.95, -1.16],
            [-0.87, 1.11, -0.83, -0.95, 0.94, 1.07, 0.91],
        ]),
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
        ],
    )


def test_exhaustive_search_hard_golay():
    code = komm.GolayCode()
    decoder = komm.ExhaustiveSearchDecoder(code, input_type="hard")
    for w in range(code.length + 1):
        for _ in range(10):
            r = np.zeros(23, dtype=int)
            error_locations = np.random.choice(23, w, replace=False)
            r[error_locations] ^= 1
            u_hat = decoder.decode(r)
            if w <= 3:  # Golay code can correct up to 3 errors.
                assert np.array_equal(u_hat, np.zeros(12, dtype=int))
            else:  # Golay code cannot correct more than 3 errors.
                assert not np.array_equal(u_hat, np.zeros(12, dtype=int))
