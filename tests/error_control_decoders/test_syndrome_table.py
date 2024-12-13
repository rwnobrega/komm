import numpy as np
import pytest

import komm


def test_syndrome_table_hamming():
    code = komm.HammingCode(3)
    decoder = komm.SyndromeTableDecoder(code)
    np.testing.assert_array_equal(
        decoder([
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


@pytest.mark.parametrize("w", range(23))
@pytest.mark.parametrize("execution_number", range(10))
def test_syndrome_table_golay(w, execution_number):
    code = komm.GolayCode()
    decoder = komm.SyndromeTableDecoder(code)
    r = np.zeros(23, dtype=int)
    error_locations = np.random.choice(23, w, replace=False)
    r[error_locations] ^= 1
    u_hat = decoder(r)
    if w <= 3:  # Golay code can correct up to 3 errors.
        assert np.array_equal(u_hat, np.zeros(12, dtype=int))
    else:  # Golay code cannot correct more than 3 errors.
        assert not np.array_equal(u_hat, np.zeros(12, dtype=int))