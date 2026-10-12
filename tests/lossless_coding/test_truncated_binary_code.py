from re import escape, findall

import numpy as np
import pytest

import komm
import komm.abc


def test_truncated_binary_basic():
    # Wikipedia example with M = 5
    code = komm.TruncatedBinaryCode(5)
    codewords = [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 1)]
    assert code.codewords == codewords
    for x, codeword in enumerate(codewords):
        assert code.encode_single(x) == codeword
        assert code.decode_single(iter(codeword)) == x


def test_truncated_binary_wikipedia():
    # Wikipedia example with M = 10
    code = komm.TruncatedBinaryCode(10)
    assert code.encode_single(5) == (1, 0, 1)
    assert code.encode_single(6) == (1, 1, 0, 0)
    assert code.encode_single(9) == (1, 1, 1, 1)


@pytest.mark.parametrize("k", range(1, 8))
def test_truncated_binary_power_of_two(k):
    # Reduces to fixed-length binary code
    code = komm.TruncatedBinaryCode(2**k)
    labeling = komm.NaturalLabeling(k)
    for x in range(2**k):
        assert np.array_equal(code.encode_single(x), labeling.indices_to_bits(x))
    assert code.lengths == [k] * 2**k


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100])
def test_truncated_binary_round_trip(M, rng):
    code = komm.TruncatedBinaryCode(M)
    for _ in range(10):
        message = rng.integers(0, M, 100)
        assert np.array_equal(code.decode(code.encode(message)), message)


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100])
def test_truncated_binary_kraft_equality(M):
    # The code is complete
    code = komm.TruncatedBinaryCode(M)
    assert code.kraft_parameter() == 1.0


@pytest.mark.parametrize("M", [2, 5, 16, 100])
def test_truncated_binary_boundary_invariant(M):
    code = komm.TruncatedBinaryCode(M)
    for x in [0, M // 2, M - 1]:
        for tail in [(), (0,), (1,), (1, 0, 1, 1, 0)]:
            bits = iter(code.encode_single(x) + tail)
            assert code.decode_single(bits) == x
            assert tuple(bits) == tail


def test_truncated_binary_empty():
    code = komm.TruncatedBinaryCode(5)
    assert code.encode([]).size == 0
    assert code.decode([]).size == 0


@pytest.mark.parametrize(
    "y, message",
    [
        ([], "input contains an incomplete codeword"),
        ([1, 1], "input contains an incomplete codeword"),
        ([1, 7, 0], "invalid bit in input: 7"),
    ],
)
def test_truncated_binary_decode_single_invalid(y, message):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match=message):
        code.decode_single(iter(y))


@pytest.mark.parametrize(
    "y, message",
    [
        ([0, 1, 1], "input contains an incomplete codeword"),
        ([1, 7, 0], "elements of 'input' must be in [0:2)"),
        ([[0, 1], [1, 0]], "'input' must be a 1D-array"),
    ],
)
def test_truncated_binary_decode_invalid(y, message):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match=escape(message)):
        code.decode(y)


@pytest.mark.parametrize("x", [-1, 5, 10])
def test_truncated_binary_encode_single_invalid(x):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match=escape("'symbol' must be in [0:5)")):
        code.encode_single(x)


@pytest.mark.parametrize(
    "x, message",
    [
        ([0, 5], "elements of 'input' must be in [0:5)"),
        ([[0, 1], [2, 3]], "'input' must be a 1D-array"),
    ],
)
def test_truncated_binary_encode_invalid(x, message):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(ValueError, match=escape(message)):
        code.encode(x)


@pytest.mark.parametrize("x", [2.0, 2.5])
def test_truncated_binary_encode_single_not_integer(x):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(TypeError, match="'symbol' must be an integer"):
        code.encode_single(x)


@pytest.mark.parametrize("M", [-1, 0, 1])
def test_truncated_binary_invalid_cardinality(M):
    with pytest.raises(ValueError, match="'cardinality' must be at least 2"):
        komm.TruncatedBinaryCode(M)


def test_truncated_binary_not_integer_code():
    assert not isinstance(komm.TruncatedBinaryCode(5), komm.abc.IntegerCode)


def test_truncated_binary_docstring_table():
    doc = str(komm.TruncatedBinaryCode.__doc__)
    lines = [line.strip() for line in doc.splitlines()]
    rows = [line.strip("|").split("|") for line in lines if line.startswith("|")]
    header, body = rows[0], rows[2:]
    cardinalities = [int(findall(r"\d+", cell)[0]) for cell in header[1:]]
    codes = [komm.TruncatedBinaryCode(M) for M in cardinalities]
    for row in body:
        x = int(row[0].strip(" $"))
        for code, cell in zip(codes, row[1:], strict=True):
            codeword = tuple(int(bit) for bit in cell.strip(" `"))
            if x < code.cardinality:
                assert code.encode_single(x) == codeword
            else:
                assert codeword == ()


@pytest.mark.parametrize("M", range(2, 65))
def test_truncated_binary_uniform_huffman(M):
    code = komm.TruncatedBinaryCode(M)
    huffman = komm.HuffmanCode([1 / M] * M, assignment="canonical")
    assert code.codewords == huffman.codewords


@pytest.mark.parametrize("method", ["encode", "decode"])
def test_truncated_binary_not_integer(method):
    code = komm.TruncatedBinaryCode(5)
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        getattr(code, method)([0.5, 1.0])


def test_truncated_binary_large_cardinality():
    # Coding does not build the table
    M = 10**18
    code = komm.TruncatedBinaryCode(M)
    x = [0, M // 2, M - 1]
    assert np.array_equal(code.decode(code.encode(x)), x)


def test_truncated_binary_repr():
    assert repr(komm.TruncatedBinaryCode(5)) == "TruncatedBinaryCode(cardinality=5)"
