from collections import Counter
from functools import cache
from itertools import product
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_pmf
from ..types import Array1D

if TYPE_CHECKING:
    from .FixedToVariableCode import FixedToVariableCode

Word = tuple[int, ...]


def is_prefix_of(w1: Word, w2: Word) -> bool:
    return len(w1) <= len(w2) and w2[: len(w1)] == w1


def is_prefix_free(words: list[Word]) -> bool:
    words = [w for w in words if len(w) > 0]  # Ignore empty words
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
) -> npt.NDArray[np.integer]:
    output: list[int] = []
    i = 0
    for j in range(len(input)):
        key = tuple(map(int, input[i : j + 1]))
        if key in dictionary:
            output.extend(dictionary[key])
            i = j + 1

    if i == len(input):
        return np.asarray(output)
    elif not allow_incomplete:
        raise ValueError("input contains invalid word")

    remainder = tuple(map(int, input[i:]))
    for key, value in dictionary.items():
        if is_prefix_of(remainder, key):
            output.extend(value)
            return np.asarray(output)

    raise ValueError("input contains invalid word")


def lexicographical_code(lengths: npt.ArrayLike) -> list[Word]:
    r"""
    Generates the lexicographical prefix-free symbol code based on the given lengths.

    Parameters:
        lengths: A list where the index is the symbol and the value is its codeword length.

    Returns:
        codewords: A list where the index is the symbol and the value is the bit tuple for that symbol. Symbols with zero length receive an empty tuple.
    """
    lengths = np.asarray(lengths, dtype=int)

    if not lengths.ndim == 1:
        raise ValueError("'lengths' must be a 1D-array")
    if not np.all(lengths >= 0):
        raise ValueError("'lengths' must be non-negative")

    l_max = lengths.max()
    counts = Counter(lengths)
    integers = [0] * (l_max + 1)
    for l in range(1, l_max + 1):
        integers[l] = (integers[l - 1] + counts[l - 1]) * 2

    codewords: list[Word] = [()] * lengths.size
    for x, l in enumerate(lengths):
        codewords[x] = integer_to_symbols(integers[l], base=2, width=l)
        integers[l] += 1

    return codewords


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
        integer = integer * base + symbol
    return integer


def create_code_from_lengths(
    pmf: npt.NDArray[np.floating],
    source_block_size: int,
    code_lengths_func,
) -> dict[Word, Word]:
    """
    Helper function to create a code mapping from a pmf and a code lengths function.
    
    This reduces duplication between FanoCode and ShannonCode.
    """
    from functools import reduce

    extended_pmf = reduce(np.multiply.outer, [pmf] * source_block_size)
    lengths = code_lengths_func(extended_pmf.ravel())
    codewords = lexicographical_code(lengths)
    enc_mapping: dict[Word, Word] = {}
    for x, c in zip(np.ndindex(extended_pmf.shape), codewords):
        enc_mapping[x] = c
    return enc_mapping


class PrefixFreePMFCode:
    """
    Base class for prefix-free codes based on a probability mass function.
    
    This base class consolidates common functionality for codes like
    FanoCode, ShannonCode, and similar PMF-based prefix-free codes.
    """

    def __init__(
        self,
        pmf: npt.ArrayLike,
        source_block_size: int,
        code_lengths_func: Callable[[Array1D[np.floating]], Array1D[np.integer]],
    ):
        # Import here to avoid circular import
        from .FixedToVariableCode import FixedToVariableCode

        self.pmf = validate_pmf(pmf)
        if not source_block_size >= 1:
            raise ValueError("'source_block_size' must be at least 1")
        FixedToVariableCode.__init__(
            self,
            source_cardinality=self.pmf.size,
            target_cardinality=2,
            source_block_size=source_block_size,
            enc_mapping=create_code_from_lengths(
                self.pmf, source_block_size, code_lengths_func
            ),
        )

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"source_block_size={self.source_block_size}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cache
    def is_uniquely_decodable(self) -> bool:
        return True

    @cache
    def is_prefix_free(self) -> bool:
        return True
