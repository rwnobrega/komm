from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NoReturn, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def tqdm_iter(iterable: Iterable[T], desc: str, **kwargs: Any) -> "tqdm[T]":
    return tqdm(iterable, desc=desc, delay=2.5, **kwargs)


def tqdm_bar(total: int, desc: str, **kwargs: Any) -> "tqdm[NoReturn]":
    return tqdm(total=total, desc=desc, delay=2.5, **kwargs)


def tqdm_chunked(items: Sequence[T], desc: str, chunk_size: int = 2**16) -> Iterator[T]:
    # Update the bar per chunk, for speed
    with tqdm_bar(len(items), desc) as pbar:
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            yield from chunk
            pbar.update(len(chunk))
