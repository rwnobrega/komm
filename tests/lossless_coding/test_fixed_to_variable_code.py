from re import escape

import numpy as np
import pytest

import komm

test_data = [
    {
        "source_cardinality": 5,
        "target_cardinality": 2,
        "source_block_size": 1,
        "codewords": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)],
    },
    {  # [Hay04, Example 9.3]
        "source_cardinality": 5,
        "target_cardinality": 2,
        "source_block_size": 1,
        "codewords": [(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)],
    },
    {
        "source_cardinality": 2,
        "target_cardinality": 2,
        "source_block_size": 2,
        "codewords": [(0, 0), (0, 1), (1, 0), (1, 1, 1)],
    },
    {  # [CT06, Example 5.6.1]
        "source_cardinality": 5,
        "target_cardinality": 2,
        "source_block_size": 1,
        "codewords": [(0, 1), (1, 0), (1, 1), (0, 0, 0), (0, 0, 1)],
    },
    {  # [CT06, Example 5.6.2]
        "source_cardinality": 5,
        "target_cardinality": 3,
        "source_block_size": 1,
        "codewords": [(1,), (2,), (0, 0), (0, 1), (0, 2)],
    },
    {  # [CT06, Example 5.6.3]
        "source_cardinality": 7,
        "target_cardinality": 3,
        "source_block_size": 1,
        "codewords": [(1,), (2,), (0, 1), (0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)],
    },
]


@pytest.mark.parametrize(
    "code_parameters",
    test_data,
)
def test_init(code_parameters):
    source_cardinality, target_cardinality, source_block_size, codewords = (
        code_parameters.values()
    )
    code = komm.FixedToVariableCode.from_codewords(
        codewords=codewords,
        source_cardinality=source_cardinality,
    )
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.source_block_size == source_block_size
    assert code.codewords == codewords
    assert code.lengths == [len(codeword) for codeword in codewords]


def test_enc_mapping_order():
    enc_mapping = {(1, 1): (1, 1, 1), (0, 0): (0,), (1, 0): (1, 1, 0), (0, 1): (1, 0)}
    code = komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    assert list(code.enc_mapping) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert code.codewords == [(0,), (1, 0), (1, 1, 0), (1, 1, 1)]
    assert code.lengths == [1, 2, 3, 3]
    assert np.isclose(code.rate([0.9, 0.1]), 0.645)


@pytest.mark.parametrize(
    "args, message",
    [
        ((1, 2, 1), "'source_cardinality' must be at least 2 (got 1)"),
        ((2, 1, 1), "'target_cardinality' must be at least 2 (got 1)"),
        ((2, 2, 0), "'source_block_size' must be a positive integer (got 0)"),
    ],
)
def test_invalid_parameters(args, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.FixedToVariableCode(*args, enc_mapping={(0,): (0,), (1,): (1,)})


def test_integer_parameters():
    enc_mapping = {(0,): (0,), (1,): (1,)}
    code = komm.FixedToVariableCode(np.int64(2), 2, 1, enc_mapping)  # type: ignore
    assert type(code.source_cardinality) is int


@pytest.mark.parametrize(
    "args, message",
    [
        ((2.0, 2, 1), "'source_cardinality' must be an integer (got float)"),
        ((2, 2.0, 1), "'target_cardinality' must be an integer (got float)"),
        ((2, 2, 1.0), "'source_block_size' must be an integer (got float)"),
    ],
)
def test_parameters_not_integer(args, message):
    with pytest.raises(TypeError, match=escape(message)):
        komm.FixedToVariableCode(*args, enc_mapping={(0,): (0,), (1,): (1,)})


@pytest.mark.parametrize(
    "enc_mapping",
    [
        {(0,): (0,), (1,): (1, 0)},  # Missing word
        {(0,): (0,), (1,): (1, 0), (2,): (1, 1, 0), (3,): (1, 1, 1)},  # Extra word
        {(0,): (0,), (1,): (1, 0), (3,): (1, 1)},  # Symbol out of range
        {(0,): (0,), (1,): (1, 0), (2, 0): (1, 1)},  # Word of wrong length
    ],
)
def test_invalid_enc_mapping_domain(enc_mapping):
    with pytest.raises(ValueError, match="'enc_mapping': invalid domain"):
        komm.FixedToVariableCode(3, 2, 1, enc_mapping)


@pytest.mark.parametrize(
    "enc_mapping",
    [
        {(0,): (0,), (1,): (1, 0), (2,): (1, 2)},  # Symbol out of range
        {(0,): (0,), (1,): (1, 0), (2,): (1, -1)},  # Negative symbol
    ],
)
def test_invalid_enc_mapping_codomain(enc_mapping):
    with pytest.raises(ValueError, match="'enc_mapping': invalid co-domain"):
        komm.FixedToVariableCode(3, 2, 1, enc_mapping)


def test_empty_codeword():
    with pytest.raises(ValueError, match="'enc_mapping': codewords must be non-empty"):
        komm.FixedToVariableCode(3, 2, 1, {(0,): (), (1,): (1, 0), (2,): (1, 1)})


@pytest.mark.parametrize(
    "enc_mapping, message",
    [
        ({(0,): (0,), (2,): (1,)}, "'enc_mapping': invalid domain"),
        ({(0,): (), (1,): (1,)}, "'enc_mapping': codewords must be non-empty"),
    ],
)
def test_from_enc_mapping_invalid(enc_mapping, message):
    with pytest.raises(ValueError, match=message):
        komm.FixedToVariableCode.from_enc_mapping(enc_mapping)


@pytest.mark.parametrize(
    "codewords, source_cardinality, message",
    [
        ([(), (0, 1)], None, "elements of 'codewords' must be non-empty"),
        ([(0,), (1,), (0, 1)], 1, "'source_cardinality' must be at least 2"),
        ([(0,), (1, 0), (1, 1)], 2, "must be a power of source cardinality 2 (got 3)"),
    ],
)
def test_from_codewords_invalid(codewords, source_cardinality, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.FixedToVariableCode.from_codewords(
            codewords,
            source_cardinality=source_cardinality,
        )


@pytest.mark.parametrize(
    "source_cardinality, codewords",
    [
        (3, [(0,), (0, 0), (1, 0)]),
        (3, [(0, 0), (0,), (1, 0)]),
        (3, [(0, 0), (1, 0), (0,)]),
    ],
)
def test_decode_not_uniquely_decodable(source_cardinality, codewords):
    code = komm.FixedToVariableCode.from_codewords(
        codewords=codewords,
        source_cardinality=source_cardinality,
    )
    with pytest.raises(ValueError, match="not uniquely decodable"):
        code.decode([0])


@pytest.mark.parametrize(
    "code_parameters, pmf, rate",
    [
        (test_data[0], [0.4, 0.2, 0.2, 0.1, 0.1], 3.0),
        (test_data[0], [0.2, 0.2, 0.2, 0.2, 0.2], 3.0),
        (test_data[1], [0.4, 0.2, 0.2, 0.1, 0.1], 2.2),
        (test_data[1], [0.2, 0.2, 0.2, 0.2, 0.2], 2.4),
        (test_data[2], [0.5, 0.5], 1.125),
        (test_data[2], [0.4, 0.6], 1.18),
        (test_data[3], [0.25, 0.25, 0.2, 0.15, 0.15], 2.3),
        (test_data[4], [0.25, 0.25, 0.2, 0.15, 0.15], 1.5),
        (test_data[5], [0.25, 0.25, 0.2, 0.1, 0.1, 0.1, 0.0], 1.7),
    ],
)
def test_rate(code_parameters, pmf, rate):
    source_cardinality, _, _, codewords = code_parameters.values()
    code = komm.FixedToVariableCode.from_codewords(
        codewords=codewords,
        source_cardinality=source_cardinality,
    )
    assert np.isclose(code.rate(pmf), rate)


@pytest.mark.parametrize(
    "pmf, message",
    [
        ([0.5, 0.5, 0.1], "'pmf' must sum to 1.0"),
        ([-0.4, 0.4, 1.0], "'pmf' must be non-negative"),
        ([0.5, 0.5], "'pmf' must have size 3 (got 2)"),
        ([0.25, 0.25, 0.25, 0.25], "'pmf' must have size 3 (got 4)"),
    ],
)
def test_rate_invalid_pmf(pmf, message):
    code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1)])
    with pytest.raises(ValueError, match=escape(message)):
        code.rate(pmf)


@pytest.mark.parametrize(
    "code_parameters, x, y",
    [
        (
            test_data[0],
            [3, 0, 1, 1, 1, 0, 2, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ),
        (
            test_data[1],
            [3, 0, 1, 1, 1, 0, 2, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        ),
        (
            test_data[2],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 0, 1, 0],
        ),
    ],
)
def test_encode_decode(code_parameters, x, y):
    source_cardinality, _, _, codewords = code_parameters.values()
    code = komm.FixedToVariableCode.from_codewords(
        codewords=codewords,
        source_cardinality=source_cardinality,
    )
    assert np.array_equal(code.encode(x), y)
    assert np.array_equal(code.decode(y), x)


@pytest.mark.parametrize(
    "y, message",
    [
        ([1, 1, 0, 1, 0, 1, 1, 1], "input contains invalid word"),
        ([1, 1, 0, 1, 0, 1, 3], "input contains invalid word"),
        ([1, 1, 0, 1, 0, 1], "input contains an incomplete codeword"),
    ],
)
def test_decode_invalid(y, message):
    code = komm.FixedToVariableCode.from_codewords([(0,), (1, 0), (1, 1, 0)])
    with pytest.raises(ValueError, match=message):
        code.decode(y)


@pytest.mark.parametrize("code_parameters", test_data)
def test_from_lengths(code_parameters):
    source_cardinality, target_cardinality, source_block_size, codewords = (
        code_parameters.values()
    )
    lengths = [len(codeword) for codeword in codewords]
    code = komm.FixedToVariableCode.from_lengths(
        lengths=lengths,
        source_cardinality=source_cardinality,
        target_cardinality=target_cardinality,
    )
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.source_block_size == source_block_size
    assert [len(codeword) for codeword in code.codewords] == lengths
    assert code.is_prefix_free()
    assert code.kraft_parameter() <= 1


@pytest.mark.parametrize(
    "lengths, source_cardinality, target_cardinality, message",
    [
        ([1, 1, 1], None, 2, "'lengths' must satisfy Kraft inequality"),
        ([0, 1, 2], None, 2, "elements of 'lengths' must be at least 1"),
        ([1, -1, 2], None, 2, "elements of 'lengths' must be at least 1"),
        ([[1, 2], [2, 2]], None, 2, "'lengths' must be a 1D-array"),
        ([1, 2, 2], 1, 2, "'source_cardinality' must be at least 2"),
        ([1, 2, 2], None, 1, "'target_cardinality' must be at least 2"),
        ([1, 2, 3, 3], 3, 2, "must be a power of source cardinality 3"),
    ],
)
def test_from_lengths_invalid(lengths, source_cardinality, target_cardinality, message):
    with pytest.raises(ValueError, match=message):
        komm.FixedToVariableCode.from_lengths(
            lengths,
            source_cardinality=source_cardinality,
            target_cardinality=target_cardinality,
        )


@pytest.mark.parametrize(
    "lengths, source_cardinality, target_cardinality, message",
    [
        ([1.7, 2.2, 2.9], None, 2, "'lengths' must contain only integers"),
        ([1, 2, 2], 3.0, 2, "'source_cardinality' must be an integer"),
        ([1, 2, 2], None, 2.0, "'target_cardinality' must be an integer"),
    ],
)
def test_from_lengths_not_integer(
    lengths, source_cardinality, target_cardinality, message
):
    with pytest.raises(TypeError, match=message):
        komm.FixedToVariableCode.from_lengths(
            lengths,
            source_cardinality=source_cardinality,
            target_cardinality=target_cardinality,
        )
