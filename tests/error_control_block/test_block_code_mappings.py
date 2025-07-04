import numpy as np
import pytest

import komm
import komm.abc

codes = [
    komm.BlockCode(generator_matrix=[[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]]),
    komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 1, 0]]),
    komm.CyclicCode(length=23, generator_polynomial=0b101011100011),
    komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 12),
    komm.PolarCode(mu=4, frozen=[0, 1, 2, 3, 4, 5, 6, 8, 9]),
]


@pytest.mark.parametrize("code", codes)
@pytest.mark.repeat(20)
def test_mappings_array_input(code: komm.abc.BlockCode):
    u1 = np.random.randint(0, 2, code.dimension)
    u2 = np.random.randint(0, 2, code.dimension)
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
def test_mappings_inverses(code: komm.abc.BlockCode):
    # Check that 'inverse_encode' is the inverse of 'encode'
    u = np.random.randint(0, 2, (3, 4, code.dimension))
    np.testing.assert_equal(u, code.inverse_encode(code.encode(u)))


@pytest.mark.parametrize("code", codes)
def test_mappings_invalid_dimensions(code: komm.abc.BlockCode):
    n, k = code.length, code.dimension

    # For 'encode', last dimension of 'u' should be a multiple of the code dimension
    code.encode(np.zeros((3, 4, k)))  # Correct
    code.encode(np.zeros((3, 4, 2 * k)))  # Correct
    with pytest.raises(ValueError):
        code.encode(np.zeros((3, 4, k + 1)))  # Incorrect

    # For 'inverse_encode', last dimension of 'v' should be a multiple of the code length
    code.inverse_encode(np.zeros((3, 4, n)))  # Correct
    code.inverse_encode(np.zeros((3, 4, 2 * n)))  # Correct
    with pytest.raises(ValueError):
        code.inverse_encode(np.zeros((3, 4, n + 1)))  # Incorrect

    # For 'check', last dimension of 'r' should be a multiple of the code length
    code.check(np.zeros((3, 4, n)))  # Correct
    code.check(np.zeros((3, 4, 2 * n)))  # Correct
    with pytest.raises(ValueError):
        code.check(np.zeros((3, 4, n + 1)))  # Incorrect


@pytest.mark.parametrize("code", codes)
def test_mappings_invalid_codewords(code: komm.abc.BlockCode):
    r = np.zeros(code.length)
    code.inverse_encode(r)  # Correct
    with pytest.raises(ValueError):
        r[0] = 1
        code.inverse_encode(r)  # Incorrect
