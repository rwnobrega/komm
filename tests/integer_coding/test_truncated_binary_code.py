from itertools import islice

import numpy as np
import pytest

import komm


def test_truncated_binary_basic():
    # Wikipedia example with M = 5
    code = komm.TruncatedBinaryCode(5)
    codewords = [[0, 0], [0, 1], [1, 0], [1, 1, 0], [1, 1, 1]]
    for n, codeword in enumerate(codewords):
        assert code.encode_single(n) == codeword
        assert code.decode_single(iter(codeword)) == n


def test_truncated_binary_wikipedia():
    # Wikipedia example with M = 10
    code = komm.TruncatedBinaryCode(10)
    assert code.encode_single(5) == [1, 0, 1]
    assert code.encode_single(6) == [1, 1, 0, 0]
    assert code.encode_single(9) == [1, 1, 1, 1]


@pytest.mark.parametrize("k", range(1, 8))
def test_truncated_binary_power_of_two(k):
    # Reduces to fixed-length binary code
    code = komm.TruncatedBinaryCode(2**k)
    labeling = komm.NaturalLabeling(k)
    for n in range(2**k):
        assert np.array_equal(code.encode_single(n), labeling.indices_to_bits(n))
        assert code.length(n) == k


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100])
def test_truncated_binary_round_trip(M):
    code = komm.TruncatedBinaryCode(M)
    for _ in range(10):
        message = np.random.randint(0, M, 100)
        assert np.array_equal(message, list(code.decode(code.encode(message))))


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100])
def test_truncated_binary_kraft_equality(M):
    # The code is complete
    code = komm.TruncatedBinaryCode(M)
    assert sum(2.0 ** -code.length(n) for n in range(M)) == 1.0


@pytest.mark.parametrize("M", [2, 5, 16, 100])
def test_truncated_binary_length(M):
    code = komm.TruncatedBinaryCode(M)
    for n in range(M):
        assert code.length(n) == len(code.encode_single(n))


@pytest.mark.parametrize("M", [2, 5, 16, 100])
def test_truncated_binary_boundary_invariant(M):
    code = komm.TruncatedBinaryCode(M)
    for n in [0, M // 2, M - 1]:
        for tail in [[], [0], [1], [1, 0, 1, 1, 0]]:
            bits = iter(code.encode_single(n) + tail)
            assert code.decode_single(bits) == n
            assert list(bits) == tail


def test_truncated_binary_lazy_decode():
    code = komm.TruncatedBinaryCode(10)
    message = [5, 0, 9, 2, 8]
    bits = code.encode(message)
    assert next(code.decode(bits)) == 5
    assert list(islice(code.decode(bits), 2)) == [0, 9]
    assert list(code.decode(bits)) == [2, 8]


def test_truncated_binary_empty():
    code = komm.TruncatedBinaryCode(5)
    assert list(code.encode([])) == []
    assert list(code.decode([])) == []
    with pytest.raises(StopIteration):
        next(code.decode([]))
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([]))


def test_truncated_binary_incomplete_codeword():
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([1, 1]))
    with pytest.raises(ValueError, match="incomplete codeword"):
        list(code.decode([0, 1, 1]))


def test_truncated_binary_invalid_bit():
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([1, 7, 0]))


@pytest.mark.parametrize("M", [2, 5, 16])
def test_truncated_binary_out_of_range(M):
    code = komm.TruncatedBinaryCode(M)
    for n in [-1, M, 2 * M]:
        with pytest.raises(ValueError, match="out-of-range"):
            code.encode_single(n)
        with pytest.raises(ValueError, match="out-of-range"):
            code.length(n)
    with pytest.raises(ValueError, match="out-of-range"):
        list(code.encode([0, M]))


@pytest.mark.parametrize("M", [-1, 0, 1])
def test_truncated_binary_invalid_cardinality(M):
    with pytest.raises(ValueError, match="at least 2"):
        komm.TruncatedBinaryCode(M)
