import numpy as np

import komm


def test_huffman_code_1():
    # Sayood.06, p. 47.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1])
    assert code.enc_mapping == {(0,): (1, 1), (1,): (0, 0), (2,): (1, 0), (3,): (0, 1, 1), (4,): (0, 1, 0)}
    assert code.rate(code.pmf) == 2.2


def test_huffman_code_2():
    # Sayood.06, p. 44.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1], policy="low")
    assert code.enc_mapping == {(0,): (0, 1), (1,): (1,), (2,): (0, 0, 0), (3,): (0, 0, 1, 0), (4,): (0, 0, 1, 1)}
    assert code.rate(code.pmf) == 2.2


def test_huffman_code_3():
    # Haykin.04, p. 620.
    code1 = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy="high")
    assert code1.enc_mapping == {
        (0,): (1, 1, 1),
        (1,): (1, 1, 0),
        (2,): (0, 0, 1),
        (3,): (1, 0, 1),
        (4,): (1, 0, 0),
        (5,): (0, 0, 0),
        (6,): (0, 1, 1),
        (7,): (0, 1, 0),
    }
    assert np.isclose(code1.rate(code1.pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code1.enc_mapping.values()]), 0.0)
    code2 = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy="low")
    assert code2.enc_mapping == {
        (0,): (1, 1, 0),
        (1,): (1, 1, 1),
        (2,): (0, 1),
        (3,): (1, 0, 0),
        (4,): (1, 0, 1),
        (5,): (0, 0, 0),
        (6,): (0, 0, 1, 0),
        (7,): (0, 0, 1, 1),
    }
    assert np.isclose(code2.rate(code2.pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code2.enc_mapping.values()]), 23 / 64)


def test_huffman_code_4():
    # Haykin.04, p. 620.
    code1 = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=1)
    assert np.isclose(code1.rate(code1.pmf), 1.3)
    code2 = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=2)
    assert np.isclose(code2.rate(code2.pmf), 1.1975)
