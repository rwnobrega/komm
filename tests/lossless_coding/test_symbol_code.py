import numpy as np
import pytest

import komm

test_data = [
    {
        "source_cardinality": 5,
        "target_cardinality": 2,
        "codewords": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)],
    },
    {  # [Hay04, Example 9.3]
        "source_cardinality": 5,
        "target_cardinality": 2,
        "codewords": [(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)],
    },
    {  # [CT06, Example 5.6.1]
        "source_cardinality": 5,
        "target_cardinality": 2,
        "codewords": [(0, 1), (1, 0), (1, 1), (0, 0, 0), (0, 0, 1)],
    },
    {  # [CT06, Example 5.6.2]
        "source_cardinality": 5,
        "target_cardinality": 3,
        "codewords": [(1,), (2,), (0, 0), (0, 1), (0, 2)],
    },
    {  # [CT06, Example 5.6.3]
        "source_cardinality": 7,
        "target_cardinality": 3,
        "codewords": [(1,), (2,), (0, 1), (0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)],
    },
]


@pytest.mark.parametrize("code_parameters", test_data)
def test_init(code_parameters):
    source_cardinality, target_cardinality, codewords = code_parameters.values()
    enc_mapping = dict(enumerate(codewords))
    code = komm.SymbolCode(source_cardinality, target_cardinality, enc_mapping)
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.enc_mapping == enc_mapping
    assert code.codewords == codewords
    assert code.lengths == [len(codeword) for codeword in codewords]


@pytest.mark.parametrize("code_parameters", test_data)
def test_from_enc_mapping(code_parameters):
    source_cardinality, target_cardinality, codewords = code_parameters.values()
    code = komm.SymbolCode.from_enc_mapping(dict(enumerate(codewords)))
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.codewords == codewords


@pytest.mark.parametrize("code_parameters", test_data)
def test_from_codewords(code_parameters):
    source_cardinality, target_cardinality, codewords = code_parameters.values()
    code = komm.SymbolCode.from_codewords(codewords)
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.codewords == codewords


@pytest.mark.parametrize("code_parameters", test_data)
def test_from_lengths(code_parameters):
    source_cardinality, target_cardinality, codewords = code_parameters.values()
    lengths = [len(codeword) for codeword in codewords]
    code = komm.SymbolCode.from_lengths(lengths, target_cardinality=target_cardinality)
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.lengths == lengths


@pytest.mark.parametrize("lengths", [[1, 2, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4]])
def test_from_lengths_alphabetic(lengths):
    # Non-decreasing lengths yield an alphabetic (slice) code. See [CT06, Sec. 5.7].
    code = komm.SymbolCode.from_lengths(lengths)
    assert code.codewords == sorted(code.codewords)


@pytest.mark.parametrize("code_parameters", test_data)
def test_fixed_to_variable_equivalence(code_parameters):
    # Same code as the fixed-to-variable one with k = 1.
    source_cardinality, target_cardinality, codewords = code_parameters.values()
    code = komm.SymbolCode(
        source_cardinality,
        target_cardinality,
        dict(enumerate(codewords)),
    )
    block_code = komm.FixedToVariableCode(
        source_cardinality,
        target_cardinality,
        1,
        {(x,): codeword for x, codeword in enumerate(codewords)},
    )
    assert code.codewords == block_code.codewords
    assert code.lengths == block_code.lengths
    assert code.is_uniquely_decodable() == block_code.is_uniquely_decodable()
    assert code.is_prefix_free() == block_code.is_prefix_free()
    assert code.kraft_parameter() == block_code.kraft_parameter()
    canonical = komm.SymbolCode.from_lengths(
        code.lengths,
        target_cardinality=target_cardinality,
    )
    block_canonical = komm.FixedToVariableCode.from_lengths(
        block_code.lengths,
        source_cardinality=source_cardinality,
        target_cardinality=target_cardinality,
    )
    assert canonical.codewords == block_canonical.codewords


def test_enc_mapping_order():
    code = komm.SymbolCode(3, 2, {2: (1, 1), 0: (0,), 1: (1, 0)})
    assert list(code.enc_mapping) == [0, 1, 2]
    assert code.codewords == [(0,), (1, 0), (1, 1)]
    assert code.lengths == [1, 2, 2]


def test_repr():
    enc_mapping = {0: (0,), 1: (1, 0), 2: (1, 1)}
    code = komm.SymbolCode(3, 2, enc_mapping)
    args = f"source_cardinality=3, target_cardinality=2, enc_mapping={enc_mapping}"
    assert repr(code) == f"SymbolCode({args})"


@pytest.mark.parametrize(
    "source_cardinality, target_cardinality, name",
    [
        (1, 2, "source_cardinality"),
        (2, 1, "target_cardinality"),
    ],
)
def test_invalid_cardinality(source_cardinality, target_cardinality, name):
    with pytest.raises(ValueError, match=f"'{name}' must be at least 2"):
        komm.SymbolCode(source_cardinality, target_cardinality, {0: (0,), 1: (1,)})


def test_integer_parameters():
    code = komm.SymbolCode(np.int64(2), 2, {0: (0,), 1: (1,)})  # type: ignore
    assert type(code.source_cardinality) is int


@pytest.mark.parametrize(
    "source_cardinality, target_cardinality, name",
    [
        (2.0, 2, "source_cardinality"),
        (2, 2.0, "target_cardinality"),
    ],
)
def test_cardinality_not_integer(source_cardinality, target_cardinality, name):
    with pytest.raises(TypeError, match=f"'{name}' must be an integer"):
        komm.SymbolCode(source_cardinality, target_cardinality, {0: (0,), 1: (1,)})


@pytest.mark.parametrize(
    "enc_mapping",
    [
        {0: (0,), 1: (1, 0)},  # Missing symbol
        {0: (0,), 1: (1, 0), 2: (1, 1, 0), 3: (1, 1, 1)},  # Extra symbol
        {0: (0,), 1: (1, 0), 3: (1, 1)},  # Symbol out of range
        {(0,): (0,), (1,): (1, 0), (2,): (1, 1)},  # Tuples, not integers
    ],
)
def test_invalid_enc_mapping_domain(enc_mapping):
    with pytest.raises(ValueError, match="'enc_mapping': invalid domain"):
        komm.SymbolCode(3, 2, enc_mapping)


@pytest.mark.parametrize(
    "enc_mapping",
    [
        {0: (0,), 1: (1, 0), 2: (1, 2)},  # Symbol out of range
        {0: (0,), 1: (1, 0), 2: (1, -1)},  # Negative symbol
    ],
)
def test_invalid_enc_mapping_codomain(enc_mapping):
    with pytest.raises(ValueError, match="'enc_mapping': invalid co-domain"):
        komm.SymbolCode(3, 2, enc_mapping)


def test_empty_codeword():
    with pytest.raises(ValueError, match="'enc_mapping': codewords must be non-empty"):
        komm.SymbolCode(3, 2, {0: (), 1: (1, 0), 2: (1, 1)})


@pytest.mark.parametrize(
    "enc_mapping, message",
    [
        ({0: (0,), 2: (1,)}, "'enc_mapping': invalid domain"),
        ({(0,): (0,), (1,): (1,)}, "'enc_mapping': invalid domain"),
        ({0: (), 1: (1,)}, "'enc_mapping': codewords must be non-empty"),
    ],
)
def test_from_enc_mapping_invalid(enc_mapping, message):
    with pytest.raises(ValueError, match=message):
        komm.SymbolCode.from_enc_mapping(enc_mapping)


def test_from_codewords_empty_codeword():
    with pytest.raises(ValueError, match="elements of 'codewords' must be non-empty"):
        komm.SymbolCode.from_codewords([(), (0, 1)])


@pytest.mark.parametrize(
    "lengths, target_cardinality, message",
    [
        ([1, 1, 1], 2, "'lengths' must satisfy Kraft inequality"),
        ([0, 1, 2], 2, "elements of 'lengths' must be at least 1"),
        ([[1, 2], [2, 2]], 2, "'lengths' must be a 1D-array"),
        ([1, 2, 2], 1, "'target_cardinality' must be at least 2"),
    ],
)
def test_from_lengths_invalid(lengths, target_cardinality, message):
    with pytest.raises(ValueError, match=message):
        komm.SymbolCode.from_lengths(lengths, target_cardinality=target_cardinality)


@pytest.mark.parametrize(
    "lengths, target_cardinality, message",
    [
        ([1.7, 2.2, 2.9], 2, "'lengths' must contain only integers"),
        ([1, 2, 2], 2.0, "'target_cardinality' must be an integer"),
    ],
)
def test_from_lengths_not_integer(lengths, target_cardinality, message):
    with pytest.raises(TypeError, match=message):
        komm.SymbolCode.from_lengths(lengths, target_cardinality=target_cardinality)
