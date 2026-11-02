import pytest

from komm._util.progress import tqdm_chunked


@pytest.mark.parametrize("chunk_size", [1, 3, 10, 20])
def test_tqdm_chunked(chunk_size):
    items = list(range(10))
    assert list(tqdm_chunked(items, "Testing", chunk_size)) == items


def test_tqdm_chunked_empty():
    assert list(tqdm_chunked([], "Testing")) == []
