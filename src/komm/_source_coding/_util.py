import itertools as it

Word = tuple[int, ...]


def is_prefix_free(words: list[Word]) -> bool:
    for c1, c2 in it.combinations(words, 2):
        if c1[: len(c2)] == c2 or c2[: len(c1)] == c1:
            return False
    return True


def is_uniquely_decodable(words: list[Word]) -> bool:
    # Sardinas–Patterson algorithm. See [Say06, Sec. 2.4.1].
    augmented_words = set(words)
    while True:
        dangling_suffixes = set()
        for c1, c2 in it.product(augmented_words, repeat=2):
            if c1 != c2 and c1[: len(c2)] == c2:
                dangling_suffixes.add(c1[len(c2) :])
        if any(suffix in words for suffix in dangling_suffixes):
            return False
        if all(suffix in augmented_words for suffix in dangling_suffixes):
            return True
        augmented_words |= dangling_suffixes


def parse_prefix_free(input_sequence, dictionary):
    output_sequence = []
    i = 0
    while i < len(input_sequence):
        j = 1
        while i + j <= len(input_sequence):
            try:
                key = tuple(input_sequence[i : i + j])
                output_sequence.extend(dictionary[key])
                break
            except KeyError:
                j += 1
        i += j
    return output_sequence
