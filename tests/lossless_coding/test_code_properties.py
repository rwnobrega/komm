import pytest

from komm._lossless_coding.util import is_complete, is_prefix_free, is_uniquely_parsable

test_cases = [
    {
        "words": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # [Say06, Sec. 2.4.1, Code 2]
        "words": [(0,), (1,), (0, 0), (1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": True,
    },
    {  # [Say06, Sec. 2.4.1, Code 3]
        "words": [(0,), (1, 0), (1, 1, 0), (1, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # [Say06, Sec. 2.4.1, Code 4]
        "words": [(0,), (0, 1), (0, 1, 1), (0, 1, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Say06, Sec. 2.4.1, Code 5]
        "words": [(0,), (0, 1), (1, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,  # But complete as a symbol code [MacK03, Sec. 5.2]
    },
    {  # [Say06, Sec. 2.4.1, Code 6]
        "words": [(0,), (0, 1), (1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # [CT06, Sec. 5.1, Table 5.1, Code 2]
        "words": [(0,), (0, 1, 0), (0, 1), (1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # [CT06, Sec. 5.1, Table 5.1, Code 3]
        "words": [(1, 0), (0, 0), (1, 1), (1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {  # Wikipedia example; [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(1,), (0, 1, 1), (0, 1, 1, 1, 0), (1, 1, 1, 0), (1, 0, 0, 1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # Old false negative for UD
        "words": [(0,), (0, 1), (1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {
        "words": [(0,), (0, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.2]: "Any suffix-free code is uniquely decodable"
        "words": [(0,), (0, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.3]
        "words": [(0, 0, 1, 1), (0, 1, 1), (1, 1), (1, 1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.3]
        "words": [(0, 1), (1, 0), (0, 1, 1), (1, 1, 0)],
        "uniquely_parsable": False,  # Course page is wrong: (01)(110) = (011)(10)
        "prefix_free": False,
        "complete": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(1,), (1, 0, 0, 0, 0, 0), (0, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(0, 1), (1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 1), (1, 1, 1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {
        "words": [(1, 0, 1), (0, 0, 1, 1, 1), (0, 1, 0, 0), (1,)],
        "uniquely_parsable": False,  # (101)(00111) = (1)(0100)(1)(1)(1)
        "prefix_free": False,
        "complete": False,
    },
    {  # Regression: Duplicate words
        "words": [(0, 1), (0, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # Regression: Duplicate words
        "words": [(0,), (0,), (1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # [Say06, Example 3.7.1]
        "words": [(0, 0, 0), (0, 0, 1), (0, 1), (1,)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # [Say06, Example 3.7.1]
        "words": [(0, 0, 0), (0, 1, 0), (0, 1), (1,)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": False,
    },
    {  # Basic complete binary code
        "words": [(0,), (1, 0), (1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # Incomplete binary code (missing sequences starting with 1)
        "words": [(0, 0), (0, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
    {  # Larger alphabet (ternary) with full coverage
        "words": [(0,), (1,), (2, 0), (2, 1), (2, 2)],
        "cardinality": 3,
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # Single-symbol code (always complete)
        "words": [(0,), (1,), (2,), (3,)],
        "cardinality": 4,
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # Prefix-free code but not complete
        "words": [(0, 0), (0, 1), (1, 0)],  # missing (1, 1, ...)
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
    {  # Complex case with longer sequences
        "words": [(0,), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": True,
    },
    {  # Ternary code with gaps
        "words": [(0,), (1,), (2, 0), (2, 1)],  # missing (2, 2, ...)
        "cardinality": 3,
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
    {  # Edge case: empty code
        "words": [],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
    {  # Single codeword (not complete)
        "words": [(0, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
    {  # Two extensions of a minimal word
        "words": [(0,), (0, 0), (0, 1), (1,)],
        "uniquely_parsable": False,
        "prefix_free": False,
        "complete": True,
    },
    {  # Empty words are ignored
        "words": [(), (0,)],
        "uniquely_parsable": True,
        "prefix_free": True,
        "complete": False,
    },
]


@pytest.mark.parametrize(
    "words, expected_prefix_free",
    [(case["words"], case["prefix_free"]) for case in test_cases],
)
def test_is_prefix_free(words, expected_prefix_free):
    assert is_prefix_free(words) == expected_prefix_free


@pytest.mark.parametrize(
    "words, expected_uniquely_parsable",
    [(case["words"], case["uniquely_parsable"]) for case in test_cases],
)
def test_is_uniquely_parsable(words, expected_uniquely_parsable):
    assert is_uniquely_parsable(words) == expected_uniquely_parsable


@pytest.mark.parametrize(
    "words, cardinality, expected_complete",
    [
        (case["words"], case.get("cardinality", 2), case["complete"])
        for case in test_cases
    ],
)
def test_is_complete(words, cardinality, expected_complete):
    assert is_complete(words, cardinality) == expected_complete


@pytest.mark.parametrize("cardinality", [2, 3])
def test_complete_parsable_prefix_free(cardinality, rng):
    # Complete and uniquely parsable implies prefix-free
    for _ in range(100):
        words = [()]
        for _ in range(rng.integers(1, 6)):  # Leaves of a full tree
            word = words.pop(rng.integers(len(words)))
            words += [word + (y,) for y in range(cardinality)]
        for _ in range(rng.integers(1, 4)):  # Extensions break prefix-freeness
            suffix = rng.integers(cardinality, size=rng.integers(1, 3)).tolist()
            words.append(words[rng.integers(len(words))] + tuple(suffix))
        assert not is_prefix_free(words)
        assert is_complete(words, cardinality)
        assert not is_uniquely_parsable(words)
