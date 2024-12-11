import numpy as np

import komm


def test_standard_array_mws():
    # [McWS, p. 16]
    code = komm.BlockCode(generator_matrix=[[1, 0, 1, 1], [0, 1, 0, 1]])
    sa = komm.SlepianArray(code)
    codewords = sa.row(0)
    np.testing.assert_array_equal(
        codewords, [[0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]
    )
    leaders = sa.col(0)
    np.testing.assert_array_equal(
        leaders, [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    messages = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for j, (message, codeword) in enumerate(zip(messages, codewords)):
        np.testing.assert_array_equal(komm.int_to_bits(j, 2), message)
        np.testing.assert_array_equal(code.encode(message), codeword)
    syndromes = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for i, (syndrome, leader) in enumerate(zip(syndromes, leaders)):
        np.testing.assert_array_equal(komm.int_to_bits(i, 2), syndrome)
        np.testing.assert_array_equal(code.check(leader), syndrome)
