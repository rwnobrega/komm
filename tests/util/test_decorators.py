import numpy as np
import pytest

from komm._util.decorators import blockwise, chunkwise, vectorize


def test_vectorize():
    @vectorize("Testing")
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
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        func([1, 2, 3, 4])


def test_chunkwise():
    sizes = []

    @chunkwise(3, "Testing")
    def func(rows):
        sizes.append(rows.shape[0])
        return rows[:, ::2]

    arr = np.arange(40).reshape(2, 5, 4)
    np.testing.assert_equal(func(arr), arr[..., ::2])
    assert sizes == [3, 3, 3, 1]
