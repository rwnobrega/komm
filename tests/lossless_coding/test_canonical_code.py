from re import escape

import pytest

from komm._lossless_coding.util import canonical_code, is_prefix_free


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([3], [(0, 0, 0)]),
        ([0, 1, 0], [(), (0,), ()]),
        ([1, 2, 3, 3], [(0,), (1, 0), (1, 1, 0), (1, 1, 1)]),
        ([0, 2, 2], [(), (0, 0), (0, 1)]),
        ([3, 1, 3, 2], [(1, 1, 0), (0,), (1, 1, 1), (1, 0)]),
        ([3, 3, 3, 3], [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]),
        ([2, 2, 5, 5], [(0, 0), (0, 1), (1, 0, 0, 0, 0), (1, 0, 0, 0, 1)]),
        ([2, 3, 3, 3, 3], [(0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]),
    ],
)
def test_canonical_code_basic(lengths, expected):
    got = canonical_code(lengths)
    assert got == expected
    assert is_prefix_free(got)


@pytest.mark.parametrize(
    "lengths, base, expected",
    [
        ([1, 1, 1], 3, [(0,), (1,), (2,)]),
        ([1, 2, 2, 2], 3, [(0,), (1, 0), (1, 1), (1, 2)]),
        ([1, 1, 2, 2, 3], 3, [(0,), (1,), (2, 0), (2, 1), (2, 2, 0)]),
        ([0, 1, 2], 4, [(), (0,), (1, 0)]),
        ([2, 1, 2], 5, [(1, 0), (0,), (1, 1)]),
    ],
)
def test_canonical_code_base(lengths, base, expected):
    got = canonical_code(lengths, base=base)
    assert got == expected
    assert is_prefix_free(got)


@pytest.mark.parametrize("base", range(2, 6))
@pytest.mark.parametrize("size", range(1, 30))
def test_canonical_code_random(base, size, rng):
    # Lengths of a random full base-ary tree always satisfy Kraft with equality.
    lengths = [1] * base
    for _ in range(size):
        i = rng.integers(len(lengths))
        lengths[i : i + 1] = [lengths[i] + 1] * base
    rng.shuffle(lengths)
    codewords = canonical_code(lengths, base=base)
    assert is_prefix_free(codewords)
    assert [len(c) for c in codewords] == lengths
    assert len(set(codewords)) == len(codewords)


@pytest.mark.parametrize(
    "lengths, base",
    [
        ([1, 1, 1], 2),
        ([1, 2, 2, 2], 2),
        ([2, 2, 2, 2, 2], 2),
        ([1, 1, 1, 1], 3),
        ([1, 1, 2, 2, 2, 2, 2, 2, 2], 3),
    ],
)
def test_canonical_code_kraft(lengths, base):
    with pytest.raises(ValueError, match="must satisfy Kraft inequality"):
        canonical_code(lengths, base=base)


@pytest.mark.parametrize(
    "lengths, base, message",
    [
        ([[1, 2], [3, 4]], 2, "'lengths' must be a 1D-array"),
        ([-1, 0, 1], 2, "elements of 'lengths' must be at least 0"),
        ([1, 2, 2], 1, "'base' must be at least 2 (got 1)"),
    ],
)
def test_canonical_code_invalid(lengths, base, message):
    with pytest.raises(ValueError, match=escape(message)):
        canonical_code(lengths, base=base)


@pytest.mark.parametrize(
    "lengths, base, message",
    [
        ([1.0, 2.0, 2.0], 2, "'lengths' must contain only integers"),
        ([1, 2, 2], 2.0, "'base' must be an integer (got float)"),
    ],
)
def test_canonical_code_not_integer(lengths, base, message):
    with pytest.raises(TypeError, match=escape(message)):
        canonical_code(lengths, base=base)
