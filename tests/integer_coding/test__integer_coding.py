from itertools import islice

import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.UnaryCode(),
        komm.EliasGammaCode(),
        komm.EliasDeltaCode(),
        komm.FibonacciCode(),
    ],
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize("n", range(1, 100))
def test_integer_coding_constants(code: komm.abc.IntegerCode, n: int):
    for r in range(10):
        message = np.full(shape=(r,), fill_value=n)
        assert np.array_equal(message, list(code.decode(code.encode(message))))


def test_integer_coding_random(code: komm.abc.IntegerCode):
    for _ in range(10):
        message = np.random.randint(1, 100, 100)
        assert np.array_equal(message, list(code.decode(code.encode(message))))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_length(code: komm.abc.IntegerCode, n: int):
    assert code.length(n) == len(code.encode_single(n))


def test_integer_coding_lazy_decode(code: komm.abc.IntegerCode):
    message = [5, 1, 9, 2, 8]
    bits = code.encode(message)
    assert next(code.decode(bits)) == 5
    assert list(islice(code.decode(bits), 2)) == [1, 9]
    assert list(code.decode(bits)) == [2, 8]


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 63, 64, 1000])
def test_integer_coding_boundary_invariant(code: komm.abc.IntegerCode, n: int):
    if not isinstance(code, komm.UnaryCode):
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
