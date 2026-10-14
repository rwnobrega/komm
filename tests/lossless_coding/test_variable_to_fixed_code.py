from re import escape

import numpy as np
import pytest

import komm


def test_lengths():
    code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (0, 1)])
    assert code.lengths == [1, 1, 2]
    assert code.lengths == [len(x) for x in code.sourcewords]


def test_dec_mapping_order():
    dec_mapping = {(1, 1): (1,), (0, 0): (0, 0), (0, 1): (0, 1)}
    code = komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    assert list(code.dec_mapping) == [(0, 0), (0, 1), (1, 1)]
    assert code.sourcewords == [(0, 0), (0, 1), (1,)]
    assert code.lengths == [2, 2, 1]


@pytest.mark.parametrize(
    "args, message",
    [
        ((1, 2, 1), "'target_cardinality' must be at least 2 (got 1)"),
        ((2, 1, 1), "'source_cardinality' must be at least 2 (got 1)"),
        ((2, 2, 0), "'target_block_size' must be a positive integer (got 0)"),
    ],
)
def test_invalid_parameters(args, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.VariableToFixedCode(*args, dec_mapping={(0,): (0,), (1,): (1,)})


def test_integer_parameters():
    dec_mapping = {(0,): (0,), (1,): (1,)}
    code = komm.VariableToFixedCode(np.int64(2), 2, 1, dec_mapping)  # type: ignore
    assert type(code.target_cardinality) is int


@pytest.mark.parametrize(
    "args, message",
    [
        ((2.0, 2, 1), "'target_cardinality' must be an integer (got float)"),
        ((2, 2.0, 1), "'source_cardinality' must be an integer (got float)"),
        ((2, 2, 1.0), "'target_block_size' must be an integer (got float)"),
    ],
)
def test_parameters_not_integer(args, message):
    with pytest.raises(TypeError, match=escape(message)):
        komm.VariableToFixedCode(*args, dec_mapping={(0,): (0,), (1,): (1,)})


@pytest.mark.parametrize(
    "sourcewords, target_cardinality, message",
    [
        ([(), (0, 1)], 2, "elements of 'sourcewords' must be non-empty"),
        ([(0,), (1,), (0, 1)], 1, "'target_cardinality' must be at least 2"),
    ],
)
def test_from_sourcewords_invalid(sourcewords, target_cardinality, message):
    with pytest.raises(ValueError, match=message):
        komm.VariableToFixedCode.from_sourcewords(
            sourcewords,
            target_cardinality=target_cardinality,
        )


@pytest.mark.parametrize(
    "dec_mapping",
    [
        {(0, 0): (0,), (2, 1): (1,)},  # Symbol out of range
        {(0, 0): (0,), (1,): (1,)},  # Word of wrong length
    ],
)
def test_invalid_dec_mapping_domain(dec_mapping):
    with pytest.raises(ValueError, match="'dec_mapping': invalid domain"):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


@pytest.mark.parametrize(
    "dec_mapping",
    [
        {(0, 0): (0,), (0, 1): (0, 2)},  # Symbol out of range
        {(0, 0): (0,), (0, 1): (0, -1)},  # Negative symbol
    ],
)
def test_invalid_dec_mapping_codomain(dec_mapping):
    with pytest.raises(ValueError, match="'dec_mapping': invalid co-domain"):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_empty_sourceword():
    with pytest.raises(
        ValueError, match="'dec_mapping': sourcewords must be non-empty"
    ):
        komm.VariableToFixedCode(2, 2, 2, {(0, 0): (0,), (0, 1): ()})


@pytest.mark.parametrize(
    "dec_mapping, message",
    [
        ({(0, 0): (0,), (1,): (1,)}, "'dec_mapping': invalid domain"),
        ({(0,): (), (1,): (1,)}, "'dec_mapping': sourcewords must be non-empty"),
    ],
)
def test_from_dec_mapping_invalid(dec_mapping, message):
    with pytest.raises(ValueError, match=message):
        komm.VariableToFixedCode.from_dec_mapping(dec_mapping)


def test_rate():
    code = komm.VariableToFixedCode.from_sourcewords(
        [(0, 0, 0), (0, 0, 1), (0, 1), (1,)], target_cardinality=2
    )
    assert np.isclose(code.rate([2 / 3, 1 / 3]), 18 / 19)


@pytest.mark.parametrize(
    "pmf, message",
    [
        ([0.5, 0.6], "'pmf' must sum to 1.0"),
        ([-0.4, 1.4], "'pmf' must be non-negative"),
        ([1.0], "'pmf' must have size 2 (got 1)"),
        ([0.2, 0.3, 0.5], "'pmf' must have size 2 (got 3)"),
    ],
)
def test_rate_invalid_pmf(pmf, message):
    code = komm.VariableToFixedCode.from_sourcewords(
        [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    with pytest.raises(ValueError, match=escape(message)):
        code.rate(pmf)


def test_encode_decode():
    # [Say06, Example 3.7.1]
    code = komm.VariableToFixedCode.from_sourcewords(
        [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    x = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    y = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    assert np.array_equal(code.encode(x), y)
    assert np.array_equal(code.decode(y), x)


def test_encode_decode_empty():
    code = komm.VariableToFixedCode.from_sourcewords(
        [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    for output in [code.encode([]), code.decode([])]:
        assert output.shape == (0,)
        assert np.issubdtype(output.dtype, np.integer)


@pytest.mark.parametrize(
    "sourcewords, x, message",
    [
        ([(0,), (1, 0), (1, 1)], [0, 7], "elements of 'input' must be in [0:2)"),
        ([(0,), (1, 0), (1, 1)], [[0, 1], [1, 0]], "'input' must be a 1D-array"),
        ([(0, 0, 0), (0, 1, 0), (0, 1), (1,)], [0], "code is not complete"),
        ([(0,), (1,), (0, 1)], [1, 0, 0, 0], "code is not uniquely encodable"),
    ],
)
def test_encode_invalid(sourcewords, x, message):
    code = komm.VariableToFixedCode.from_sourcewords(sourcewords)
    with pytest.raises(ValueError, match=escape(message)):
        code.encode(x)


@pytest.mark.parametrize("x", [[0.0, 1.0], [0.5, 1.0]])
def test_encode_not_integer(x):
    code = komm.VariableToFixedCode.from_sourcewords([(0,), (1, 0), (1, 1)])
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        code.encode(x)


@pytest.mark.parametrize(
    "sourcewords, y, message",
    [
        ([(0,), (1,), (2,)], [1, 1, 0, 0, 1], "multiple of block size 2 (got 5)"),
        ([(0,), (1,), (2,)], [0, 0, 1, 1], "input contains invalid word"),
        ([(0,), (1,), (2,)], [0, 2], "elements of 'input' must be in [0:2)"),
        ([(0,), (1,), (2,)], [[0, 0], [1, 0]], "'input' must be a 1D-array"),
        ([(0, 0), (0, 1)], [0, 0, 0, 1], "code is not complete"),
        ([(0,), (1,), (0, 1)], [0, 0, 0, 1], "code is not uniquely encodable"),
    ],
)
def test_decode_invalid(sourcewords, y, message):
    code = komm.VariableToFixedCode.from_sourcewords(sourcewords)
    with pytest.raises(ValueError, match=escape(message)):
        code.decode(y)


@pytest.mark.parametrize("y", [[1.0, 0.0], [0.5, 1.0]])
def test_decode_not_integer(y):
    code = komm.VariableToFixedCode.from_sourcewords([(0,), (1,), (2,)])
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        code.decode(y)
