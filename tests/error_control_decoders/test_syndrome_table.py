import numpy as np
import pytest

import komm


def test_syndrome_table_hamming():
    code = komm.HammingCode(3)
    decoder = komm.SyndromeTableDecoder(code)
    np.testing.assert_equal(
        decoder.decode([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 0],
        ]),
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ],
    )


@pytest.mark.parametrize("systematic", [False, True])
def test_syndrome_table_cyclic(systematic, rng):
    # BCH (15, 7), which corrects up to 2 errors.
    code = komm.CyclicCode(
        length=15,
        generator_polynomial=0b111010001,
        systematic=systematic,
    )
    decoder = komm.SyndromeTableDecoder(code)
    u = rng.integers(0, 2, (100, code.dimension))
    e = np.zeros((100, code.length), dtype=int)
    for row in e:
        row[rng.choice(code.length, 2, replace=False)] = 1
    np.testing.assert_equal(decoder.decode(code.encode(u) ^ e), u)


def test_syndrome_table_golay(rng):
    code = komm.GolayCode()
    decoder = komm.SyndromeTableDecoder(code)
    for w in range(code.length + 1):
        for _ in range(10):
            r = np.zeros(23, dtype=int)
            error_locations = rng.choice(23, w, replace=False)
            r[error_locations] ^= 1
            u_hat = decoder.decode(r)
            if w <= 3:  # Golay code can correct up to 3 errors.
                assert np.array_equal(u_hat, np.zeros(12, dtype=int))
            else:  # Golay code cannot correct more than 3 errors.
                assert not np.array_equal(u_hat, np.zeros(12, dtype=int))
