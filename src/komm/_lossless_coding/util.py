from itertools import product

import numpy as np
import numpy.typing as npt

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


def is_uniquely_decipherable(words: list[Word]) -> bool:
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


def parse_fixed_length(
    input: npt.NDArray[np.integer],
    dictionary: dict[Word, Word],
    block_size: int,
) -> npt.NDArray[np.integer]:
    if input.size % block_size != 0:
        raise ValueError(
            "length of 'input' must be a multiple of block size"
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
    while i < len(input):
        j = 1
        while i + j <= len(input):
            try:
                key = tuple(input[i : i + j])
                output.extend(dictionary[key])
                break
            except KeyError:
                j += 1
        i += j

    if i == len(input):
        return np.asarray(output)

    if not allow_incomplete:
        raise ValueError("input contains invalid word")

    remaining = tuple(input[i:])
    for key, value in dictionary.items():
        if is_prefix_of(remaining, key):
            output.extend(value)
            return np.asarray(output)

    raise ValueError("input contains invalid word")


def is_fully_covering(words: list[Word], cardinality: int) -> bool:
    class TrieNode:
        def __init__(self):
            self.id = id(self)
            self.is_end: bool = False
            self.children: dict[int, TrieNode] = {}

    def build_trie(words: list[Word]) -> TrieNode:
        root = TrieNode()
        for w in words:
            current = root
            for symbol in w:
                if symbol not in current.children:
                    current.children[symbol] = TrieNode()
                current = current.children[symbol]
            current.is_end = True
        return root

    def check_coverage_from_node(node: TrieNode, visited: set[int]) -> bool:
        # Recursively check if all possible sequences from this node lead to valid words.
        # Uses DFS with cycle detection to handle infinite paths.

        if node.id in visited:
            return True

        visited.add(node.id)

        for symbol in range(cardinality):
            if symbol not in node.children:
                return False
            child = node.children[symbol]
            if child.is_end:
                continue
            if not check_coverage_from_node(child, visited):
                return False

        return True

    root = build_trie(words)
    return check_coverage_from_node(root, set())
