from collections import Counter
from itertools import product

import numpy as np
import numpy.typing as npt

Word = tuple[int, ...]


def is_prefix_of(w1: Word, w2: Word) -> bool:
    return len(w1) <= len(w2) and w2[: len(w1)] == w1


def is_prefix_free(words: list[Word]) -> bool:
    words = sorted(w for w in words if len(w) > 0)  # Ignore empty words
    return not any(is_prefix_of(w1, w2) for w1, w2 in zip(words, words[1:]))


def is_uniquely_parsable(words: list[Word]) -> bool:
    # Sardinas–Patterson algorithm. See [Say06, Sec. 2.4.1].
    words = [w for w in words if len(w) > 0]  # Ignore empty words
    if len(set(words)) < len(words):  # Duplicated words
        return False
    if is_prefix_free(words):  # Prefix-free implies uniquely parsable
        return True
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
    class Node:
        def __init__(self):
            self.is_end: bool = False
            self.children: dict[int, "Node"] = {}

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
        output_list: list[int] = []
        for i in range(0, len(input), block_size):
            key = tuple(map(int, input[i : i + block_size]))
            output_list.extend(dictionary[key])
        output = np.asarray(output_list)
    except KeyError:
        raise ValueError("input contains invalid word")
    return output


def parse_prefix_free(
    input: npt.NDArray[np.integer],
    dictionary: dict[Word, Word],
    allow_incomplete: bool,
    cardinality: int = 2,
) -> npt.NDArray[np.integer]:
    # Precompute a (length, value) -> word table, where 'value' is the word
    # folded into an integer in base 'cardinality'. Parsing is then a single
    # pass which folds input symbols into an accumulator and probes the table.
    lut: dict[tuple[int, int], Word] = {}
    max_length = 0
    for key, value in dictionary.items():
        acc = 0
        for symbol in key:
            acc = cardinality * acc + symbol
        lut[len(key), acc] = value
        max_length = max(max_length, len(key))

    input = np.asarray(input)
    if input.size > 0 and (input.min() < 0 or input.max() >= cardinality):
        raise ValueError("input contains invalid word")

    output: list[int] = []
    length = acc = 0
    for symbol in input.tolist():  # Python ints are faster
        acc = cardinality * acc + symbol
        length += 1
        if (length, acc) in lut:
            output.extend(lut[length, acc])
            length = acc = 0
        elif length > max_length:
            raise ValueError("input contains invalid word")

    if length == 0:
        return np.asarray(output)
    elif not allow_incomplete:
        raise ValueError("input contains invalid word")

    remainder: list[int] = []
    for _ in range(length):
        acc, symbol = divmod(acc, cardinality)
        remainder.insert(0, symbol)
    for key, value in dictionary.items():
        if is_prefix_of(tuple(remainder), key):
            output.extend(value)
            return np.asarray(output)

    raise ValueError("input contains invalid word")


def infer_block_size(size: int, cardinality: int, name: str) -> int:
    k, power = 1, cardinality
    while power < size:
        power *= cardinality
        k += 1
    if power != size:
        raise ValueError(
            f"length of '{name}' must be a power of source cardinality"
            f" {cardinality} (got {size})"
        )
    return k


def canonical_code(lengths: npt.ArrayLike, base: int = 2) -> list[Word]:
    r"""
    Generates the canonical (lexicographical) prefix-free symbol code based on the given lengths.

    Parameters:
        lengths: A list where the index is the symbol and the value is its codeword length. Must be non-negative and satisfy the Kraft inequality (symbols with zero length are ignored).

        base: The base (i.e., the target cardinality) of the code. Must be at least 2. The default value is 2.

    Returns:
        codewords: A list where the index is the symbol and the value is the symbol tuple for that symbol. Symbols with zero length receive an empty tuple.
    """
    lengths = np.asarray(lengths, dtype=int)

    if not lengths.ndim == 1:
        raise ValueError("'lengths' must be a 1D-array")
    if not np.all(lengths >= 0):
        raise ValueError("'lengths' must be non-negative")
    if not base >= 2:
        raise ValueError("'base' must be at least 2")

    lengths_list: list[int] = lengths.tolist()
    l_max = max(lengths_list)
    counts = Counter(l for l in lengths_list if l > 0)  # Zero lengths take no budget
    integers = [0] * (l_max + 1)
    for l in range(1, l_max + 1):
        integers[l] = (integers[l - 1] + counts[l - 1]) * base

    # Kraft inequality, checked exactly in (arbitrary precision) integer arithmetic.
    # Since integers[l + 1] = (integers[l] + counts[l]) * base, a violation at any
    # length propagates to l_max, so it suffices to check the condition there.
    if integers[l_max] + counts[l_max] > base**l_max:
        raise ValueError("'lengths' must satisfy Kraft inequality")

    codewords: list[Word] = [()] * len(lengths_list)
    for x, l in enumerate(lengths_list):
        codewords[x] = integer_to_symbols(integers[l], base=base, width=l)
        integers[l] += 1

    return codewords


def find_longest_match(buffer: bytes, i: int, ss: int, l_max: int) -> tuple[int, int]:
    # Rightmost longest match of buffer[i:] starting in buffer[i - ss : i], overlap allowed.
    # Returns (p, l), with p relative to i - ss.
    # Since a match of length l implies one of length l - 1, binary search on l.
    start = i - ss
    p, lo, hi = ss - 1, 0, l_max  # p = ss - 1 when l = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        q = buffer.rfind(buffer[i : i + mid], start, i + mid - 1)
        if q >= 0:
            p, lo = q - start, mid
        else:
            hi = mid - 1
    return p, lo


def integer_to_symbols(integer: int, base: int, width: int) -> Word:
    symbols: list[int] = []
    for _ in range(width):
        integer, symbol = divmod(integer, base)
        symbols.append(symbol)
    return tuple(symbols[::-1])


def symbols_to_integer(symbols: npt.ArrayLike, base: int) -> int:
    symbols = np.asarray(symbols)
    integer = 0
    for symbol in symbols:
        # int() prevents numpy fixed-width scalars from contaminating the
        # accumulator, which must remain an arbitrary-precision Python int.
        integer = integer * base + int(symbol)
    return integer
