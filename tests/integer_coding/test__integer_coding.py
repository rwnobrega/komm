from collections.abc import Iterator
from re import escape, findall

import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.UnaryCode(),
        komm.UnaryCode(stop_bit=0),
        komm.EliasGammaCode(),
        komm.EliasDeltaCode(),
        komm.EliasOmegaCode(),
        komm.FibonacciCode(),
        komm.GolombCode(1),
        komm.GolombCode(3),
        komm.GolombCode(4),
        komm.TabooCode(2),
        komm.TabooCode(3),
        komm.TabooCode(4),
        komm.EscapeCode(2),
        komm.EscapeCode(3),
        komm.EscapeCode(4),
    ],
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize("n", range(1, 100))
def test_integer_coding_constants(code: komm.abc.IntegerCode, n: int):
    for r in range(10):
        message = np.full(shape=(r,), fill_value=n)
        assert np.array_equal(message, code.decode(code.encode(message)))


def test_integer_coding_random(code: komm.abc.IntegerCode, rng):
    for _ in range(10):
        message = rng.integers(1, 100, 100)
        assert np.array_equal(message, code.decode(code.encode(message)))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_length(code: komm.abc.IntegerCode, n: int):
    assert code.length(n) == len(code.encode_single(n))


def test_integer_coding_length_numpy(code: komm.abc.IntegerCode):
    for n in np.arange(1, 100):
        assert code.length(n) == len(code.encode_single(n))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_boundary_invariant(code: komm.abc.IntegerCode, n: int):
    if not isinstance(code, (komm.UnaryCode, komm.GolombCode, komm.EscapeCode)):
        n = n * 10**6 + 1
    for tail in [(), (0,), (1,), (1, 0, 1, 1, 0)]:
        bits = iter(code.encode_single(n) + tail)
        assert code.decode_single(bits) == n
        assert tuple(bits) == tail


def test_integer_coding_empty(code: komm.abc.IntegerCode):
    assert code.encode([]).size == 0
    assert code.decode([]).size == 0
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([]))


@pytest.mark.parametrize("n", [2, 45, 1000])
def test_integer_coding_incomplete(code: komm.abc.IntegerCode, n: int):
    codeword = code.encode_single(n)
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter(codeword[:-1]))
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode(codeword[:-1])


@pytest.mark.parametrize("integer", [0, -1])
def test_integer_coding_rejects_nonpositive(code: komm.abc.IntegerCode, integer):
    with pytest.raises(ValueError, match="'integer' must be a positive integer"):
        code.encode_single(integer)
    with pytest.raises(ValueError, match="'integer' must be a positive integer"):
        code.length(integer)


@pytest.mark.parametrize("integer", [4.0, 4.5])
def test_integer_coding_rejects_float(code: komm.abc.IntegerCode, integer):
    with pytest.raises(TypeError, match="'integer' must be an integer"):
        code.encode_single(integer)
    with pytest.raises(TypeError, match="'integer' must be an integer"):
        code.length(integer)


def test_integer_coding_kraft(code: komm.abc.IntegerCode):
    total = sum(2.0 ** -code.length(n) for n in range(1, 10**4))
    assert 0.5 < total <= 1.0


def test_integer_coding_composition(code: komm.abc.IntegerCode):
    unary = komm.UnaryCode()
    message = [9, 2, 5]
    bits = iter(unary.encode_single(len(message)) + tuple(code.encode(message)))
    num = unary.decode_single(bits)
    assert [code.decode_single(bits) for _ in range(num)] == message
    assert list(bits) == []


def test_integer_coding_instance(code: komm.abc.IntegerCode):
    assert isinstance(code, komm.abc.IntegerCode)


@pytest.mark.parametrize(
    "cls",
    [
        komm.UnaryCode,
        komm.GolombCode,
        komm.EscapeCode,
        komm.EliasGammaCode,
        komm.EliasDeltaCode,
        komm.EliasOmegaCode,
        komm.FibonacciCode,
        komm.TabooCode,
    ],
)
def test_integer_coding_docstring_table(cls):
    lines = [line.strip() for line in str(cls.__doc__).splitlines()]
    rows = [line.strip("|").split("|") for line in lines if line.startswith("|")]
    header, body = rows[0], rows[2:]
    # Parameters come from column headers
    codes = [cls(*map(int, findall(r"\d+", cell))) for cell in header[1:]]
    for row in body:
        n = int(row[0].strip(" $"))
        for code, cell in zip(codes, row[1:], strict=True):
            codeword = tuple(int(bit) for bit in cell.strip(" `").replace(" ", ""))
            assert code.encode_single(n) == codeword


@pytest.mark.parametrize("n", [1, 45])
def test_integer_coding_decode_single_not_integer(code: komm.abc.IntegerCode, n: int):
    bits = iter([float(bit) for bit in code.encode_single(n)])
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        code.decode_single(bits)  # type: ignore


def test_integer_coding_decode_single_numpy(code: komm.abc.IntegerCode):
    message = [9, 2, 5]
    bits: Iterator[np.integer] = iter(code.encode(message))
    assert [code.decode_single(bits) for _ in message] == message
    assert list(bits) == []


@pytest.mark.parametrize(
    "x, message",
    [
        ([1, 0, 2], "elements of 'input' must be at least 1"),
        ([[1, 2], [3, 4]], "'input' must be a 1D-array"),
    ],
)
def test_integer_coding_encode_invalid(code: komm.abc.IntegerCode, x, message):
    with pytest.raises(ValueError, match=escape(message)):
        code.encode(x)


@pytest.mark.parametrize(
    "y, message",
    [
        ([1, 2, 0], "elements of 'input' must be in [0:2)"),
        ([[0, 1], [1, 0]], "'input' must be a 1D-array"),
    ],
)
def test_integer_coding_decode_invalid(code: komm.abc.IntegerCode, y, message):
    with pytest.raises(ValueError, match=escape(message)):
        code.decode(y)


@pytest.mark.parametrize("method", ["encode", "decode"])
def test_integer_coding_not_integer(code: komm.abc.IntegerCode, method):
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        getattr(code, method)([0.5, 1.0])
