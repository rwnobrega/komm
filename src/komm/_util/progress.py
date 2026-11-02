from collections.abc import Iterator, Sequence
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def tqdm_chunked(items: Sequence[T], desc: str, chunk_size: int = 2**16) -> Iterator[T]:
    # Update the bar per chunk, for speed
    with tqdm(total=len(items), desc=desc, delay=2.5) as pbar:
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            yield from chunk
            pbar.update(len(chunk))
