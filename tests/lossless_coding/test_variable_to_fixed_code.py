import numpy as np
import pytest

import komm


def test_invalid_target_cardinality():
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(1, 2, 1, {(0,): (0,)})


def test_invalid_source_cardinality():
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 1, 1, {(0,): (0,), (1,): (0, 0)})


def test_invalid_dec_mapping_domain_1():
    dec_mapping: dict = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1),
        (1, 1): (1,),
    }
    komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    dec_mapping[(2, 1)] = dec_mapping.pop((0, 1))
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_invalid_dec_mapping_domain_2():
    dec_mapping: dict = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1),
        (1, 1): (1,),
    }
    komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    dec_mapping[(0,)] = dec_mapping.pop((0, 1))
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_invalid_dec_mapping_codomain_1():
    dec_mapping: dict = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1),
        (1, 1): (1,),
    }
    komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    dec_mapping[(0, 1)] = (0, 0, 2)
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_invalid_dec_mapping_codomain_2():
    dec_mapping: dict = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1),
        (1, 1): (1,),
    }
    komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    dec_mapping[(0, 1)] = ()
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_non_injective_dec_mapping():
    dec_mapping: dict = {
        (0, 0): (0, 0, 0),
        (0, 1): (0, 0, 1),
        (1, 0): (0, 1),
        (1, 1): (1,),
    }
    komm.VariableToFixedCode(2, 2, 2, dec_mapping)
    dec_mapping[(1, 1)] = (0, 1)
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(2, 2, 2, dec_mapping)


def test_rate():
    code = komm.VariableToFixedCode.from_sourcewords(
        2, [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    assert np.isclose(code.rate([2 / 3, 1 / 3]), 18 / 19)


@pytest.mark.parametrize(
    "pmf",
    [[0.5, 0.5, 0.1], [-0.4, 0.4, 1.0]],
)
def test_rate_invalid_pmf(pmf):
    code = komm.VariableToFixedCode.from_sourcewords(
        2, [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    with pytest.raises(ValueError):
        code.rate(pmf)


def test_encoding_decoding():
    # [Say06, Example 3.7.1]
    code = komm.VariableToFixedCode.from_sourcewords(
        2, [(0, 0, 0), (0, 0, 1), (0, 1), (1,)]
    )
    x = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    y = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    assert np.array_equal(code.encode(x), y)
    assert np.array_equal(code.decode(y), x)


def test_encoding_not_fully_covering():
    # [Say06, Example 3.7.1]
    code = komm.VariableToFixedCode.from_sourcewords(
        2, [(0, 0, 0), (0, 1, 0), (0, 1), (1,)]
    )
    with pytest.raises(ValueError):  # Code is not fully covering
        code.encode([0])
