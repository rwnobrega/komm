import pytest

import numpy as np
import komm


b2h = komm.util.binarray2hexstr


def test_lfsr_sequence():
    lfsr = komm.LFSRSequence(feedback_poly=komm.BinaryPolynomial.from_exponents([5, 2, 0]))
    assert np.array_equal(lfsr.sequence, [0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1])
    lfsr = komm.LFSRSequence(feedback_poly=0b10000001001)
    assert b2h(lfsr.sequence[:200]) == '0029c2be33e317f3c10ff3e8cd4ddb2e5a28ef55db079896dc'


@pytest.mark.parametrize('num_states', range(2, 13))
def test_lfsr_mls(num_states):
    lfsr = komm.LFSRSequence.maximum_length_sequence(num_states)
    assert num_states == lfsr.feedback_poly.degree
    assert lfsr.length == 2**lfsr.feedback_poly.degree - 1

    seq = (-1)**lfsr.sequence
    shifts = np.arange(lfsr.length)
    cyclic_acorr = np.empty_like(shifts, dtype=np.float)
    for (i, ell) in enumerate(shifts):
        cyclic_acorr[i] = np.dot(seq, np.roll(seq, ell)) / lfsr.length
    cyclic_accor_expected = np.full_like(shifts, fill_value=-1/lfsr.length, dtype=np.float)
    cyclic_accor_expected[0] = 1.0

    assert(np.allclose(cyclic_acorr, cyclic_accor_expected))
