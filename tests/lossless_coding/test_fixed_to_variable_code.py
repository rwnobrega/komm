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
    with pytest.raises(ValueError, match="'source_cardinality' must be"):
        komm.FixedToVariableCode(1, 2, 1, {(0,): (1, 1)})


def test_invalid_target_cardinality():
    with pytest.raises(ValueError, match="'target_cardinality' must be"):
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
    with pytest.raises(ValueError, match="enc_mapping': invalid domain"):
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
    with pytest.raises(ValueError, match="enc_mapping': invalid domain"):
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
    with pytest.raises(ValueError, match="enc_mapping': invalid domain"):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_enc_mapping_domain_4():
    enc_mapping: dict = {
        (0,): (0,),
        (1,): (1, 0, 0),
        (2,): (1, 1),
    }
    komm.FixedToVariableCode(3, 2, 1, enc_mapping)
    enc_mapping[(0, 0)] = (1, 0, 1)
    with pytest.raises(ValueError, match="enc_mapping': invalid domain"):
        komm.FixedToVariableCode(3, 2, 2, enc_mapping)


def test_invalid_enc_mapping_codomain():
    enc_mapping: dict = {
        (0, 0): (0,),
        (0, 1): (1, 0, 0),
        (1, 0): (1, 1),
        (1, 1): (1, 0, 1),
    }
    komm.FixedToVariableCode(2, 2, 2, enc_mapping)
    enc_mapping[(0, 1)] = (1, 0, 2)
    with pytest.raises(ValueError, match="enc_mapping': invalid co-domain"):
        komm.FixedToVariableCode(2, 2, 2, enc_mapping)


def test_invalid_codewords_empty():
    with pytest.raises(ValueError, match="codewords' must be non-empty"):
        komm.FixedToVariableCode.from_codewords(2, [(), (0, 1)])


def test_from_codewords_invalid_source_cardinality():
    with pytest.raises(ValueError, match="'source_cardinality' must be at least 2"):
        komm.FixedToVariableCode.from_codewords(1, [(0,), (1,), (0, 1)])


@pytest.mark.parametrize(
    "source_cardinality, codewords",
    [
        (3, [(0,), (0, 0), (1, 0)]),
        (3, [(0, 0), (0,), (1, 0)]),
        (3, [(0, 0), (1, 0), (0,)]),
    ],
)
def test_decoding_not_uniquely_decodable(source_cardinality, codewords):
    code = komm.FixedToVariableCode.from_codewords(source_cardinality, codewords)
    with pytest.raises(ValueError, match="not uniquely decodable"):
        code.decode([0])


@pytest.mark.parametrize(
    "code_parameters, pmf, rate",
    # fmt: off
    [
        (test_data[0], [0.4, 0.2, 0.2, 0.1, 0.1], 3.0),
        (test_data[0], [0.2, 0.2, 0.2, 0.2, 0.2], 3.0),
        (test_data[1], [0.4, 0.2, 0.2, 0.1, 0.1], 2.2),  # [Hay04, Example 9.3]
        (test_data[1], [0.2, 0.2, 0.2, 0.2, 0.2], 2.4),
        (test_data[2], [0.5, 0.5], 1.125),
        (test_data[2], [0.4, 0.6], 1.18),
        (test_data[3], [0.25, 0.25, 0.2, 0.15, 0.15], 2.3),  # [CT06, Example 5.6.1]
        (test_data[4], [0.25, 0.25, 0.2, 0.15, 0.15], 1.5),  # [CT06, Example 5.6.2]
        (test_data[5], [0.25, 0.25, 0.2, 0.1, 0.1, 0.1, 0.0], 1.7),  # [CT06, Example 5.6.3]
    ],
    # fmt: on
)
def test_rate(code_parameters, pmf, rate):
    source_cardinality, _, _, codewords = code_parameters.values()
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
    with pytest.raises(ValueError, match="pmf must"):
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


def test_invalid_decoding_input():
    code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1, 0), (1, 1, 0)])
    assert np.array_equal(code.decode([1, 1, 0, 1, 0]), [2, 1])
    assert np.array_equal(code.decode([1, 1, 0, 1, 0, 0]), [2, 1, 0])
    assert np.array_equal(code.decode([1, 1, 0, 1, 0, 1, 0]), [2, 1, 1])
    assert np.array_equal(code.decode([1, 1, 0, 1, 0, 1, 1, 0]), [2, 1, 2])
    with pytest.raises(ValueError, match="contains invalid word"):
        code.decode([1, 1, 0, 1, 0, 1, 1, 1])  # Invalid codeword
    with pytest.raises(ValueError, match="contains invalid word"):
        code.decode([1, 1, 0, 1, 0, 1])  # Incomplete codeword
    with pytest.raises(ValueError, match="contains invalid word"):
        code.decode([1, 1, 0, 1, 0, 1, 3])  # Invalid symbol


@pytest.mark.parametrize("code_parameters", test_data)
def test_from_lengths(code_parameters):
    source_cardinality, target_cardinality, source_block_size, codewords = (
        code_parameters.values()
    )
    lengths = [len(codeword) for codeword in codewords]
    code = komm.FixedToVariableCode.from_lengths(
        source_cardinality, lengths, target_cardinality
    )
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality
    assert code.source_block_size == source_block_size
    assert [len(codeword) for codeword in code.codewords] == lengths
    assert code.is_prefix_free()
    assert code.kraft_parameter() <= 1


@pytest.mark.parametrize("pmf", [[0.4, 0.3, 0.2, 0.1], [0.5, 0.25, 0.125, 0.125]])
def test_from_lengths_canonical_huffman(pmf):
    # The canonical code with the Huffman lengths is an optimal code as well.
    huffman = komm.HuffmanCode(pmf)
    lengths = [len(codeword) for codeword in huffman.codewords]
    code = komm.FixedToVariableCode.from_lengths(len(pmf), lengths)
    assert code.is_prefix_free()
    np.testing.assert_allclose(code.rate(pmf), huffman.rate(pmf))


@pytest.mark.parametrize("lengths", [[1, 2, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4]])
def test_from_lengths_alphabetic(lengths):
    # Non-decreasing lengths yield an alphabetic (slice) code. See [CT06, Sec. 5.7].
    code = komm.FixedToVariableCode.from_lengths(2, lengths)
    assert code.codewords == sorted(code.codewords)


def test_from_lengths_invalid():
    with pytest.raises(ValueError, match="'source_cardinality' must be at least 2"):
        komm.FixedToVariableCode.from_lengths(1, [1, 2, 2])
    with pytest.raises(ValueError, match="'target_cardinality' must be at least 2"):
        komm.FixedToVariableCode.from_lengths(3, [1, 2, 2], 1)
    with pytest.raises(ValueError, match="'lengths' must satisfy Kraft inequality"):
        komm.FixedToVariableCode.from_lengths(3, [1, 1, 1])
    with pytest.raises(ValueError, match="'lengths' must be positive"):
        komm.FixedToVariableCode.from_lengths(3, [0, 1, 2])
    with pytest.raises(ValueError, match="'lengths' must be non-negative"):
        komm.FixedToVariableCode.from_lengths(3, [1, -1, 2])
    with pytest.raises(ValueError, match="'enc_mapping': invalid domain"):
        komm.FixedToVariableCode.from_lengths(3, [1, 2, 3, 3])
