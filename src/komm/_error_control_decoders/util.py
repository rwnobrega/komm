from typing import Any

from tqdm import tqdm


def get_pbar(total: int, algorithm: str) -> "tqdm[Any]":
    return tqdm(
        total=total,
        desc=f"Decoding with {algorithm} algorithm",
        unit="block",
        delay=2.5,
    )
