import pytest

from komm._lossless_coding.util import (
    is_fully_covering,
    is_prefix_free,
    is_uniquely_parsable,
)

test_cases = [
    {
        "words": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
    },
    {  # [Say06, Sec. 2.4.1, Code 2]
        "words": [(0,), (1,), (0, 0), (1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # [Say06, Sec. 2.4.1, Code 3]
        "words": [(0,), (1, 0), (1, 1, 0), (1, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": True,
    },
    {  # [Say06, Sec. 2.4.1, Code 4]
        "words": [(0,), (0, 1), (0, 1, 1), (0, 1, 1, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # [Say06, Sec. 2.4.1, Code 5]
        "words": [(0,), (0, 1), (1, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # [Say06, Sec. 2.4.1, Code 6]
        "words": [(0,), (0, 1), (1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # [CT06, Sec. 5.1, Table 5.1, Code 2]
        "words": [(0,), (0, 1, 0), (0, 1), (1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # [CT06, Sec. 5.1, Table 5.1, Code 3]
        "words": [(1, 0), (0, 0), (1, 1), (1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # Wikipedia example
        "words": [(1,), (0, 1, 1), (0, 1, 1, 1, 0), (1, 1, 1, 0), (1, 0, 0, 1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # Old false negative for UD
        "words": [(0,), (0, 1), (1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {
        "words": [(0,), (0, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.2]: "Any suffix-free code is uniquely decodable"
        "words": [(0,), (0, 1)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.3]
        "words": [(0, 0, 1, 1), (0, 1, 1), (1, 1), (1, 1, 1, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.3]
        "words": [(0, 1), (1, 0), (0, 1, 1), (1, 1, 0)],
        "uniquely_parsable": False,  # Course page is wrong: (01)(110) = (011)(10)
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(1,), (1, 0, 0, 0, 0, 0), (0, 0)],
        "uniquely_parsable": True,
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(0, 1), (1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 1), (1, 1, 1, 0)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {  # [Sedgewick-Wayne, Exercise 5.5.4]
        "words": [(1,), (0, 1, 1), (0, 1, 1, 1, 0), (1, 1, 1, 0), (1, 0, 0, 1, 1)],
        "uniquely_parsable": False,
        "prefix_free": False,
    },
    {
        "words": [(1, 0, 1), (0, 0, 1, 1, 1), (0, 1, 0, 0), (1,)],
        "uniquely_parsable": False,  # (101)(00111) = (1)(0100)(1)(1)(1)
        "prefix_free": False,
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
    "words, cardinality, expected",
    [
        (  # [Say06, Example 3.7.1]
            [(0, 0, 0), (0, 0, 1), (0, 1), (1,)],
            2,
            True,
        ),
        (  # [Say06, Example 3.7.1]
            [(0, 0, 0), (0, 1, 0), (0, 1), (1,)],
            2,
            False,
        ),
        (  # Basic fully covering binary code
            [(0,), (1, 0), (1, 1)],
            2,
            True,
        ),
        (  # Incomplete binary code (missing sequences starting with 1)
            [(0, 0), (0, 1)],
            2,
            False,
        ),
        (  # Larger alphabet (ternary) with full coverage
            [(0,), (1,), (2, 0), (2, 1), (2, 2)],
            3,
            True,
        ),
        (  # Single-symbol code (always fully covering)
            [(0,), (1,), (2,), (3,)],
            4,
            True,
        ),
        (  # Missing short sequences but covering long ones
            [(0, 0, 0), (0, 0, 1), (0, 1), (1,)],
            2,
            True,
        ),
        (  # Prefix-free code but not fully covering
            [(0, 0), (0, 1), (1, 0)],  # missing (1, 1, ...)
            2,
            False,
        ),
        (  # Complex case with longer sequences
            [(0,), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
            2,
            True,
        ),
        (  # Ternary code with gaps
            [(0,), (1,), (2, 0), (2, 1)],  # missing (2, 2, ...)
            3,
            False,
        ),
        (  # Edge case: empty code
            [],
            2,
            False,
        ),
        (  # Single codeword (not covering)
            [(0, 1, 1)],
            2,
            False,
        ),
    ],
)
def test_is_fully_covering(words, cardinality, expected):
    assert is_fully_covering(words, cardinality) == expected
