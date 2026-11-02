import pytest

from komm._util.progress import tqdm_bar, tqdm_chunked, tqdm_iter


def test_tqdm_iter():
    assert list(tqdm_iter(range(5), "Testing")) == [0, 1, 2, 3, 4]


def test_tqdm_bar():
    pbar = tqdm_bar(10, "Testing", initial=1)
    pbar.update(2)
    assert (pbar.total, pbar.n) == (10, 3)
    pbar.close()


@pytest.mark.parametrize("chunk_size", [1, 3, 10, 20])
def test_tqdm_chunked(chunk_size):
    items = list(range(10))
    assert list(tqdm_chunked(items, "Testing", chunk_size)) == items


def test_tqdm_chunked_empty():
    assert list(tqdm_chunked([], "Testing")) == []
