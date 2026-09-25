from io import StringIO

import numpy as np
from tqdm import tqdm

from komm._util.decorators import chunkwise


def test_chunkwise():
    pbar = tqdm(total=10, file=StringIO())
    sizes = []

    @chunkwise(3, pbar)
    def func(rows):
        sizes.append(rows.shape[0])
        return rows[:, ::2]

    arr = np.arange(40).reshape(2, 5, 4)
    np.testing.assert_equal(func(arr), arr[..., ::2])
    assert sizes == [3, 3, 3, 1]
    assert pbar.n == 10
