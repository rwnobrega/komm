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
    code = komm.FixedToVariableCode.from_codewords(source_cardinality, codewords)
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.source_block_size == source_block_size
    assert code.codewords == codewords


def test_invalid_source_cardinality():
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(1, 2, 1, {(0,): (1, 1)})


def test_invalid_target_cardinality():
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 1, 1, {(0,): (0,), (1,): (0, 0)})


def test_invalid_enc_mapping_domain_1():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    del enc_mapping[(1, 0)]
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_enc_mapping_domain_2():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(2, 1)] = enc_mapping.pop((0, 1))
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_enc_mapping_domain_3():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(0,)] = enc_mapping.pop((0, 1))
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_enc_mapping_domain_4():
    enc_mapping: dict = {(0,): (0,), (1,): (1, 0, 0), (2,): (1, 1)}
    komm.FixedToVariableCode(3, 2, 1, enc_mapping)
    enc_mapping[(0, 0)] = (1, 0, 1)
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(3, 2, 2, enc_mapping)


def test_invalid_enc_mapping_codomain_1():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(0, 1)] = (1, 0, 2)
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_enc_mapping_codomain_2():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(0, 1)] = ()
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_non_injective_enc_mapping():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(1, 1)] = (1, 1)
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


@pytest.mark.parametrize(
    "source_cardinality, codewords",
    [
        (3, [(0,), (0, 0), (1, 0)]),
        (3, [(0, 0), (0,), (1, 0)]),
        (3, [(0, 0), (1, 0), (0,)]),
    ],
)
def test_decoding_not_prefix_free(source_cardinality, codewords):
    code = komm.FixedToVariableCode.from_codewords(source_cardinality, codewords)
    with pytest.raises(ValueError):
        code.decode([0, 0, 0, 0])


@pytest.mark.parametrize(
    "code_parameters, pmf, rate",
    [
        (test_data[0], [0.4, 0.2, 0.2, 0.1, 0.1], 3.0),
        (test_data[0], [0.2, 0.2, 0.2, 0.2, 0.2], 3.0),
        (test_data[1], [0.4, 0.2, 0.2, 0.1, 0.1], 2.2),  # [Hay04, Example 9.3]
        (test_data[1], [0.2, 0.2, 0.2, 0.2, 0.2], 2.4),
        (test_data[2], [0.5, 0.5], 1.125),
        (test_data[2], [0.4, 0.6], 1.18),
        (test_data[3], [0.25, 0.25, 0.2, 0.15, 0.15], 2.3),  # [CT06, Example 5.6.1]
        (test_data[4], [0.25, 0.25, 0.2, 0.15, 0.15], 1.5),  # [CT06, Example 5.6.2]
        (
            test_data[5],
            [0.25, 0.25, 0.2, 0.1, 0.1, 0.1, 0.0],
            1.7,
        ),  # [CT06, Example 5.6.3]
    ],
)
def test_rate(code_parameters, pmf, rate):
    source_cardinality, target_cardinality, source_block_size, codewords = (
        code_parameters.values()
    )
    code = komm.FixedToVariableCode.from_codewords(source_cardinality, codewords)
    assert np.isclose(code.rate(pmf), rate)


@pytest.mark.parametrize(
    "pmf",
    [
        [0.5, 0.5, 0.1],
        [-0.4, 0.4, 1.0],
    ],
)
def test_rate_invalid_pmf(pmf):
    code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
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
def test_encoding_decoding(code_parameters, x, y):
    source_cardinality, _, _, codewords = code_parameters.values()
    code = komm.FixedToVariableCode.from_codewords(source_cardinality, codewords)
    assert np.array_equal(code.encode(x), y)
    assert np.array_equal(code.decode(y), x)
