from itertools import islice
from re import findall

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
        assert np.array_equal(message, list(code.decode(code.encode(message))))


def test_integer_coding_random(code: komm.abc.IntegerCode, rng):
    for _ in range(10):
        message = rng.integers(1, 100, 100)
        assert np.array_equal(message, list(code.decode(code.encode(message))))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_length(code: komm.abc.IntegerCode, n: int):
    assert code.length(n) == len(code.encode_single(n))


def test_integer_coding_length_numpy(code: komm.abc.IntegerCode):
    for n in np.arange(1, 100):
        assert code.length(n) == len(code.encode_single(n))


def test_integer_coding_lazy_decode(code: komm.abc.IntegerCode):
    message = [5, 1, 9, 2, 8]
    bits = code.encode(message)
    assert next(code.decode(bits)) == 5
    assert list(islice(code.decode(bits), 2)) == [1, 9]
    assert list(code.decode(bits)) == [2, 8]


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_boundary_invariant(code: komm.abc.IntegerCode, n: int):
    if not isinstance(code, (komm.UnaryCode, komm.GolombCode, komm.EscapeCode)):
        n = n * 10**6 + 1
    for tail in [[], [0], [1], [1, 0, 1, 1, 0]]:
        bits = iter(code.encode_single(n) + tail)
        assert code.decode_single(bits) == n
        assert list(bits) == tail


def test_integer_coding_empty(code: komm.abc.IntegerCode):
    assert list(code.encode([])) == []
    assert list(code.decode([])) == []
    with pytest.raises(StopIteration):
        next(code.decode([]))
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([]))


def test_integer_coding_incomplete(code: komm.abc.IntegerCode):
    for n in [2, 45, 1000]:
        codeword = code.encode_single(n)
        with pytest.raises(ValueError, match="incomplete codeword"):
            code.decode_single(iter(codeword[:-1]))
        with pytest.raises(ValueError, match="incomplete codeword"):
            list(code.decode(codeword[:-1]))


@pytest.mark.parametrize("message", [[0], [-1], [1, 0, 2]])
def test_integer_coding_rejects_nonpositive(code: komm.abc.IntegerCode, message):
    with pytest.raises(ValueError, match="non-positive"):
        list(code.encode(message))
    with pytest.raises(ValueError, match="non-positive"):
        code.length(min(message))


@pytest.mark.parametrize("integer", [4.0, 4.5])
def test_integer_coding_rejects_float(code: komm.abc.IntegerCode, integer):
    with pytest.raises(TypeError):
        code.encode_single(integer)
    with pytest.raises(TypeError):
        code.length(integer)


def test_integer_coding_kraft(code: komm.abc.IntegerCode):
    total = sum(2.0 ** -code.length(n) for n in range(1, 10**4))
    assert 0.5 < total <= 1.0


def test_integer_coding_composition(code: komm.abc.IntegerCode):
    unary = komm.UnaryCode()
    message = [9, 2, 5]
    bits = iter(unary.encode_single(len(message)) + list(code.encode(message)))
    num = unary.decode_single(bits)
    assert list(islice(code.decode(bits), num)) == message
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
            codeword = [int(bit) for bit in cell.strip(" `").replace(" ", "")]
            assert code.encode_single(n) == codeword
