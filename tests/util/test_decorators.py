from io import StringIO

import numpy as np
import pytest
from tqdm import tqdm

from komm._util.decorators import blockwise, chunkwise, vectorize, with_pbar


def test_vectorize():
    @vectorize
    def func(row):
        assert row.ndim == 1
        return row[::2]

    arr = np.arange(40).reshape(2, 5, 4)
    np.testing.assert_equal(func(arr), arr[..., ::2])
    np.testing.assert_equal(func(arr[0, 0]), arr[0, 0, ::2])


def test_blockwise():
    @blockwise(3)
    def func(blocks):
        assert blocks.shape[-1] == 3
        return blocks.sum(axis=-1, keepdims=True)

    np.testing.assert_equal(func([1, 2, 3, 4, 5, 6]), [6, 15])
    arr = np.arange(24).reshape(2, 2, 6)
    np.testing.assert_equal(func(arr), arr.reshape(2, 2, 2, 3).sum(axis=-1))
    with pytest.raises(ValueError):
        func([1, 2, 3, 4])


def test_chunkwise():
    sizes = []

    @chunkwise(3)
    def func(rows):
        sizes.append(rows.shape[0])
        return rows[:, ::2]

    arr = np.arange(40).reshape(2, 5, 4)
    np.testing.assert_equal(func(arr), arr[..., ::2])
    assert sizes == [3, 3, 3, 1]


def test_with_pbar():
    pbar = tqdm(total=4, file=StringIO())

    @with_pbar(pbar)
    def func(arr):
        return 2 * arr

    np.testing.assert_equal(func(np.array([1, 2])), [2, 4])
    assert pbar.n == 1
    np.testing.assert_equal(func(np.ones((3, 2))), np.full((3, 2), 2))
    assert pbar.n == 4
