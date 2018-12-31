import numpy as np
import komm


def test_fixed_to_variable_code():
    code1 = komm.FixedToVariableCode([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)])
    code2 = komm.FixedToVariableCode([(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)])
    x = [3, 0, 1, 1, 1, 0, 2, 0]
    y1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y2 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]

    assert np.array_equal(code1.encode(x), y1)
    assert np.array_equal(code1.decode(y1), x)
    assert np.array_equal(code2.encode(x), y2)
    assert np.array_equal(code2.decode(y2), x)
    assert np.isclose(code1.rate(pmf), 3.0)
    assert np.isclose(code2.rate(pmf), 2.2)


def test_huffman_code():
    # Sayood.06, p. 47.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1])
    assert code.enc_mapping == {(0,): (1, 1), (1,): (0, 0), (2,): (1, 0), (3,): (0, 1, 1), (4,): (0, 1, 0)}
    assert code.rate(code.pmf) == 2.2

    # Sayood.06, p. 44.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1], policy='low')
    assert code.enc_mapping == {(0,): (0, 1), (1,): (1,), (2,): (0, 0, 0), (3,): (0, 0, 1, 0), (4,): (0, 0, 1, 1)}
    assert code.rate(code.pmf) == 2.2

    # Haykin.04, p. 620.
    code1 = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy='high')
    assert code1.enc_mapping == {(0,): (1, 1, 1), (1,): (1, 1, 0), (2,): (0, 0, 1), (3,): (1, 0, 1), (4,): (1, 0, 0), (5,): (0, 0, 0), (6,): (0, 1, 1), (7,): (0, 1, 0)}
    assert np.isclose(code1.rate(code1.pmf), 3.0)
    #assert np.isclose(code1.variance(code1.pmf), 0.0)
    code2 = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy='low')
    assert code2.enc_mapping == {(0,): (1, 1, 0), (1,): (1, 1, 1), (2,): (0, 1), (3,): (1, 0, 0), (4,): (1, 0, 1), (5,): (0, 0, 0), (6,): (0, 0, 1, 0), (7,): (0, 0, 1, 1)}
    assert np.isclose(code2.rate(code2.pmf), 3.0)
    #assert np.isclose(code2.variance(code2.pmf), 0.4)

    # Haykin.04, p. 620.
    code1 = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=1)
    assert np.isclose(code1.rate(code1.pmf), 1.3)
    code2 = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=2)
    assert np.isclose(code2.rate(code2.pmf), 1.1975)


def test_variable_to_fixed_code():
    # Sayood.06, p. 69.
    code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
    assert np.array_equal(code.encode([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    assert np.array_equal(code.decode([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]), [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    assert np.isclose(code.rate([2/3, 1/3]), 18 / 19)


def test_tunstall_code():
    # Sayood.06, p. 71.
    code = komm.TunstallCode([0.6, 0.3, 0.1], code_block_size=3)
    assert code.enc_mapping == {(0, 0, 0): (0, 0, 0), (0, 0, 1): (0, 0, 1), (0, 0, 2): (0, 1, 0), (0, 1): (0, 1, 1), (0, 2): (1, 0, 0), (1,): (1, 0, 1), (2,): (1, 1, 0)}
    assert np.isclose(code.rate(code.pmf), 3 / 1.96)
