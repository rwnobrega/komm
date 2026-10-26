import numpy as np
import pytest

import komm
import komm.abc

codes = [
    komm.BlockCode(generator_matrix=[[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]]),
    komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]]),
    komm.CyclicCode(length=23, generator_polynomial=0b101011100011),
    komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 12),
    komm.TerminatedConvolutionalCode(
        komm.ConvolutionalCode([[0o7, 0o5]]),
        num_blocks=12,
        puncturing_matrix=[[1, 1], [1, 0]],
    ),
    komm.PolarCode(mu=4, frozen=[0, 1, 2, 3, 4, 5, 6, 8, 9]),
    komm.ReedSolomonCode(mu=3, delta=5),
]


@pytest.mark.parametrize("code", codes)
@pytest.mark.repeat(20)
def test_mappings_array_input(code: komm.abc.BlockCode, rng):
    u1 = rng.integers(0, 2, code.dimension)
    u2 = rng.integers(0, 2, code.dimension)
    v1 = code.encode(u1)
    v2 = code.encode(u2)
    # Single sequence with two codewords
    u = np.concatenate([u1, u2])
    v = np.concatenate([v1, v2])
    np.testing.assert_equal(code.encode(u), v)
    np.testing.assert_equal(code.inverse_encode(v), u)
    np.testing.assert_equal(code.check(v), np.zeros(2 * code.redundancy))
    # 2D array of single codewords
    u = np.array([u1, u2])
    v = np.array([v1, v2])
    np.testing.assert_equal(code.encode(u), v)
    np.testing.assert_equal(code.inverse_encode(v), u)
    np.testing.assert_equal(code.check(v), np.zeros((2, code.redundancy)))


@pytest.mark.parametrize("code", codes)
@pytest.mark.repeat(20)
def test_mappings_inverses(code: komm.abc.BlockCode, rng):
    # Check that 'inverse_encode' is the inverse of 'encode'
    u = rng.integers(0, 2, (3, 4, code.dimension))
    np.testing.assert_equal(u, code.inverse_encode(code.encode(u)))


@pytest.mark.parametrize("code", codes)
def test_mappings_invalid_dimensions(code: komm.abc.BlockCode):
    n, k = code.length, code.dimension

    # For 'encode', last dimension of 'u' should be a multiple of the code dimension
    code.encode(np.zeros((3, 4, k), dtype=int))  # Correct
    code.encode(np.zeros((3, 4, 2 * k), dtype=int))  # Correct
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        code.encode(np.zeros((3, 4, k + 1), dtype=int))  # Incorrect

    # For 'inverse_encode', last dimension of 'v' should be a multiple of the code length
    code.inverse_encode(np.zeros((3, 4, n), dtype=int))  # Correct
    code.inverse_encode(np.zeros((3, 4, 2 * n), dtype=int))  # Correct
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        code.inverse_encode(np.zeros((3, 4, n + 1), dtype=int))  # Incorrect

    # For 'check', last dimension of 'r' should be a multiple of the code length
    code.check(np.zeros((3, 4, n), dtype=int))  # Correct
    code.check(np.zeros((3, 4, 2 * n), dtype=int))  # Correct
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        code.check(np.zeros((3, 4, n + 1), dtype=int))  # Incorrect


@pytest.mark.parametrize("code", codes)
def test_mappings_invalid_codewords(code: komm.abc.BlockCode):
    r = np.zeros(code.length, dtype=int)
    code.inverse_encode(r)  # Correct
    r[0] = 1
    with pytest.raises(ValueError, match="one or more inputs in 'v' are not valid"):
        code.inverse_encode(r)  # Incorrect


mappings = [("encode", "dimension"), ("inverse_encode", "length"), ("check", "length")]


@pytest.mark.parametrize("code", codes)
@pytest.mark.parametrize("method, size", mappings)
@pytest.mark.parametrize("bit", [2, -1])
def test_mappings_invalid_input(code: komm.abc.BlockCode, method, size, bit):
    x = np.zeros(getattr(code, size), dtype=int)
    x[0] = bit
    with pytest.raises(ValueError, match="elements of 'input' must be 0 or 1"):
        getattr(code, method)(x)


@pytest.mark.parametrize("code", codes)
@pytest.mark.parametrize("method, size", mappings)
def test_mappings_input_not_integer(code: komm.abc.BlockCode, method, size):
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        getattr(code, method)(np.zeros(getattr(code, size)))


@pytest.mark.parametrize("code", codes)
@pytest.mark.repeat(20)
def test_project_word_with_erasures_agrees_on_codewords(code: komm.abc.BlockCode, rng):
    u = rng.integers(0, 2, (3, 4, code.dimension))
    v = code.encode(u)
    np.testing.assert_equal(
        code.project_word_with_erasures(v),
        code.project_word(v),
    )


@pytest.mark.parametrize("code", codes)
def test_project_word_with_erasures_all_erased(code: komm.abc.BlockCode):
    v = np.full((2, 3, 2 * code.length), 2)
    np.testing.assert_equal(
        code.project_word_with_erasures(v),
        np.full((2, 3, 2 * code.dimension), 2),
    )


def test_project_word_with_erasures_systematic():
    code = komm.HammingCode(3)
    v = [[2, 0, 1, 1, 2, 2, 0], [0, 2, 1, 2, 0, 1, 2]]
    np.testing.assert_equal(
        code.project_word_with_erasures(v),
        [[2, 0, 1, 1], [0, 2, 1, 2]],
    )


def test_project_word_with_erasures_generic():
    code = komm.BlockCode(generator_matrix=[[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]])
    G_r_inv = code.generator_matrix_right_inverse
    v = np.array([1, 2, 1, 0, 1])
    expected = np.where(G_r_inv[v == 2].any(axis=0), 2, ((v == 1) @ G_r_inv) % 2)
    np.testing.assert_equal(code.project_word_with_erasures(v), expected)


def trellis_words(trellis: komm.Trellis) -> list[tuple[int, ...]]:
    # Output words of all paths
    paths = [(s, ()) for s in range(trellis[0].num_states)]
    for section in trellis:
        paths = [
            (section.transitions[s, x], word + (section.outputs[s, x],))
            for s, word in paths
            for x in range(section.num_input_symbols)
            if section.transitions[s, x] >= 0
        ]
    return sorted(word for _, word in paths)


@pytest.mark.parametrize("code", codes)
def test_minimal_trellis_paths(code: komm.abc.BlockCode):
    expected = sorted(tuple(word) for word in code.codewords())
    assert trellis_words(code.minimal_trellis()) == expected


def test_minimal_trellis_same_code():
    # Single parity-check code, three ways
    trellises = [
        komm.BlockCode(generator_matrix=[[1, 1, 0], [0, 1, 1]]).minimal_trellis(),
        komm.SystematicBlockCode(parity_submatrix=[[1], [1]]).minimal_trellis(),
        komm.CyclicCode(length=3, generator_polynomial=0b11).minimal_trellis(),
    ]
    assert repr(trellises[1]) == repr(trellises[0])
    assert repr(trellises[2]) == repr(trellises[0])
