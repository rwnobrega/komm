import itertools as it

Word = tuple[int, ...]


def is_prefix_free(words: list[Word]) -> bool:
    for c1, c2 in it.permutations(words, 2):
        if c1[: len(c2)] == c2:
            return False
    return True


def is_uniquely_decodable(words: list[Word]) -> bool:
    # Sardinas–Patterson algorithm. See [Say06, Sec. 2.4.1].
    augmented_words = set(words)
    while True:
        dangling_suffixes = set()
        for c1, c2 in it.permutations(augmented_words, 2):
            if c1[: len(c2)] == c2:
                dangling_suffixes.add(c1[len(c2) :])
        if dangling_suffixes & set(words):
            return False
        if dangling_suffixes <= augmented_words:
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
