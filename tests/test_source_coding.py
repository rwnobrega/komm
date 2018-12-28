import numpy as np
import komm


def test_prefix_code():
    mapping_1 = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)]
    mapping_2 = [(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)]

    prefix_code_1 = komm.PrefixCode(mapping_1)
    prefix_code_2 = komm.PrefixCode(mapping_2)
    x = [3, 0, 1, 1, 1, 0, 2, 0]
    y_1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_2 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]

    assert np.array_equal(prefix_code_1.encode(x), y_1)
    assert np.array_equal(prefix_code_1.decode(y_1), x)
    assert np.array_equal(prefix_code_2.encode(x), y_2)
    assert np.array_equal(prefix_code_2.decode(y_2), x)

    assert np.isclose(prefix_code_1.average_length(pmf), 3.0)
    assert np.isclose(prefix_code_2.average_length(pmf), 2.2)


def test_huffman_code():
    # Sayood.06, p. 47.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1])
    assert code.mapping == [(1, 1), (0, 0), (1, 0), (0, 1, 1), (0, 1, 0)]
    assert code.average_length(code.pmf) == 2.2

    # Sayood.06, p. 44.
    code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1], policy='low')
    assert code.mapping == [(0, 1), (1,), (0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)]
    assert code.average_length(code.pmf) == 2.2


    # Haykin.04, p. 620
    code = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy='high')
    assert code.mapping == [(1, 1, 1), (1, 1, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0), (0, 1, 1), (0, 1, 0)]
    assert code.average_length(code.pmf) == 3.0
    assert code.variance(code.pmf) == 0.0
    code = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy='low')
    assert code.mapping == [(1, 1, 0), (1, 1, 1), (0, 1), (1, 0, 0), (1, 0, 1), (0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)]
    assert code.average_length(code.pmf) == 3.0
    assert code.variance(code.pmf) == 0.4

