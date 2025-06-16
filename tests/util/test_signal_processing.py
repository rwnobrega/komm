import numpy as np
import pytest

import komm


def test_sampling_rate_expand_basic():
    x = [1, 2, 3]
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=2),
        [1, 0, 2, 0, 3, 0],
    )
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=3, offset=1),
        [0, 1, 0, 0, 2, 0, 0, 3, 0],
    )


def test_sampling_rate_expand_complex():
    x = [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=2),
        [[1 + 1j, 0, 2 + 2j, 0], [3 + 3j, 0, 4 + 4j, 0]],
    )
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=3, offset=1),
        [[0, 1 + 1j, 0, 0, 2 + 2j, 0], [0, 3 + 3j, 0, 0, 4 + 4j, 0]],
    )


def test_sampling_rate_expand_axis():
    x = [[1, 2], [3, 4]]
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=2, axis=0),
        [[1, 2], [0, 0], [3, 4], [0, 0]],
    )
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=2, axis=1),
        [[1, 0, 2, 0], [3, 0, 4, 0]],
    )
    assert np.array_equal(
        komm.sampling_rate_expand(x, factor=3, offset=1, axis=1),
        [[0, 1, 0, 0, 2, 0], [0, 3, 0, 0, 4, 0]],
    )


def test_sampling_rate_expand_invalid_parameters():
    with pytest.raises(ValueError, match="should be a positive integer"):
        komm.sampling_rate_expand([1, 2, 3], factor=0)
    with pytest.raises(ValueError, match="should satisfy 0 <= offset < factor"):
        komm.sampling_rate_expand([1, 2, 3], factor=3, offset=3)


def test_sampling_rate_compress_basic():
    x = [1, 2, 3, 4, 5, 6]
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=2),
        [1, 3, 5],
    )
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=3, offset=1),
        [2, 5],
    )


def test_sampling_rate_compress_complex():
    x = [[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j]]
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=2),
        [[1 + 1j, 3 + 3j], [5 + 5j, 7 + 7j]],
    )
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=3, offset=1),
        [[2 + 2j], [6 + 6j]],
    )


def test_sampling_rate_compress_axis():
    x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=2, axis=0),
        [[1, 2, 3, 4]],
    )
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=2, axis=1),
        [[1, 3], [5, 7]],
    )
    assert np.array_equal(
        komm.sampling_rate_compress(x, factor=3, offset=1, axis=1),
        [[2], [6]],
    )


def test_sampling_rate_compress_invalid_parameters():
    with pytest.raises(ValueError, match="should be a positive integer"):
        komm.sampling_rate_compress([1, 2, 3], factor=0)
    with pytest.raises(ValueError, match="should satisfy 0 <= offset < factor"):
        komm.sampling_rate_compress([1, 2, 3], factor=3, offset=3)
