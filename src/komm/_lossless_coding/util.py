from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.information_theory import PMF

Word = tuple[int, ...]


def is_prefix_of(w1: Word, w2: Word) -> bool:
    return len(w1) <= len(w2) and w2[: len(w1)] == w1


def is_prefix_free(words: list[Word]) -> bool:
    words = sorted(words, key=len)
    for i, w1 in enumerate(words):
        for w2 in words[i + 1 :]:
            if is_prefix_of(w1, w2):
                return False
    return True


def is_uniquely_parsable(words: list[Word]) -> bool:
    # Sardinas–Patterson algorithm. See [Say06, Sec. 2.4.1].
    augmented_words = set(words)
    while True:
        dangling_suffixes: set[Word] = set()
        for w1, w2 in product(set(words), augmented_words):
            if w1 == w2:
                continue
            if is_prefix_of(w1, w2):
                dangling_suffixes.add(w2[len(w1) :])
            elif is_prefix_of(w2, w1):
                dangling_suffixes.add(w1[len(w2) :])
        if dangling_suffixes & set(words):
            return False
        if dangling_suffixes <= augmented_words:
            return True
        augmented_words |= dangling_suffixes


def is_fully_covering(words: list[Word], cardinality: int) -> bool:
    @dataclass(eq=False)
    class Node:
        is_end: bool = False
        children: dict[int, "Node"] = field(default_factory=dict)

    # Build trie
    root = Node()
    for word in words:
        node = root
        for symbol in word:
            if symbol not in node.children:
                node.children[symbol] = Node()
            node = node.children[symbol]
        node.is_end = True

    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for symbol in range(cardinality):
            if symbol not in node.children:
                return False
            child = node.children[symbol]
            if not child.is_end and child not in visited:
                visited.add(child)
                stack.append(child)

    return True


def parse_fixed_length(
    input: npt.NDArray[np.integer],
    dictionary: dict[Word, Word],
    block_size: int,
) -> npt.NDArray[np.integer]:
    if input.size % block_size != 0:
        raise ValueError(
            "length of input must be a multiple of block size"
            f" {block_size} (got {len(input)})"
        )
    try:
        output = np.concatenate(
            [dictionary[tuple(w)] for w in input.reshape(-1, block_size)]
        )
    except KeyError:
        raise ValueError("input contains invalid word")
    return output


def parse_prefix_free(
    input: npt.NDArray[np.integer],
    dictionary: dict[Word, Word],
    allow_incomplete: bool,
) -> npt.NDArray[np.integer]:
    output: list[int] = []
    i = 0
    for j in range(len(input)):
        key = tuple(input[i : j + 1])
        if key in dictionary:
            output.extend(dictionary[key])
            i = j + 1

    if i == len(input):
        return np.asarray(output)
    elif not allow_incomplete:
        raise ValueError("input contains invalid word")

    remainder = tuple(input[i:])
    for key, value in dictionary.items():
        if is_prefix_of(remainder, key):
            output.extend(value)
            return np.asarray(output)

    raise ValueError("input contains invalid word")


def extended_probabilities(
    pmf: PMF, k: int, pbar: "tqdm[Any]"
) -> list[tuple[Word, float]]:
    probs: list[tuple[float, Word]] = []
    for u in product(range(pmf.size), repeat=k):
        pu = prod(pmf[list(u)])
        probs.append((-pu, u))
        pbar.update()
    return [(u, -minus_pu) for minus_pu, u in sorted(probs)]


def empty_mapping(cardinality: int, block_size: int) -> dict[Word, Word]:
    return {x: () for x in product(range(cardinality), repeat=block_size)}


def integer_to_symbols(integer: int, base: int, width: int) -> list[int]:
    symbols: list[int] = []
    for _ in range(width):
        integer, symbol = divmod(integer, base)
        symbols.append(symbol)
    return symbols[::-1]


def symbols_to_integer(symbols: npt.ArrayLike, base: int) -> int:
    symbols = np.asarray(symbols)
    integer = 0
    for symbol in symbols:
        integer = integer * base + symbol
    return integer
