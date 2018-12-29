import numpy as np
import komm


def test_fixed_to_variable_code():
    code1 = komm.FixedToVariableLengthCode([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)])
    code2 = komm.FixedToVariableLengthCode([(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)])
    x = [3, 0, 1, 1, 1, 0, 2, 0]
    y1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y2 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]

    assert np.array_equal(code1.encode(x), y1)
    assert np.array_equal(code1.decode(y1), x)
    assert np.array_equal(code2.encode(x), y2)
    assert np.array_equal(code2.decode(y2), x)
    assert np.isclose(code1.average_length(pmf), 3.0)
    assert np.isclose(code2.average_length(pmf), 2.2)


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
    assert np.isclose(code.average_length(code.pmf), 3.0)
    assert np.isclose(code.variance(code.pmf), 0.0)
    code = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy='low')
    assert code.mapping == [(1, 1, 0), (1, 1, 1), (0, 1), (1, 0, 0), (1, 0, 1), (0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)]
    assert np.isclose(code.average_length(code.pmf), 3.0)
    assert np.isclose(code.variance(code.pmf), 0.4)
