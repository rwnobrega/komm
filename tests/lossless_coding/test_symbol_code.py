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
