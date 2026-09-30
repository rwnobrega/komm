import numpy as np
import pytest

import komm
import komm.abc

systematic_codes = [
    komm.HammingCode(3),
    komm.GolayCode(),
    komm.SystematicBlockCode(
        parity_submatrix=[[0, 1, 1], [1, 1, 0]],
        information_set="right",
    ),
]


@pytest.mark.parametrize("code", systematic_codes)
def test_exhaustive_bitwise_hard_is_sign(code: komm.abc.BlockCode, rng):
    soft = komm.ExhaustiveBitwiseDecoder(code)
    hard = komm.ExhaustiveBitwiseDecoder(code, output_type="hard")
    li = rng.standard_normal((20, code.length))
    np.testing.assert_equal(hard.decode(li), soft.decode(li) < 0)
    np.testing.assert_equal(
        hard.decode_to_codeword(li), soft.decode_to_codeword(li) < 0
    )


@pytest.mark.parametrize("code", systematic_codes)
def test_exhaustive_bitwise_systematic(code: komm.SystematicBlockCode, rng):
    decoder = komm.ExhaustiveBitwiseDecoder(code)
    li = rng.standard_normal((20, code.length))
    np.testing.assert_allclose(
        decoder.decode(li),
        decoder.decode_to_codeword(li)[:, code.information_set],
    )


def test_exhaustive_bitwise_probabilities(rng):
    code = komm.HammingCode(3)
    decoder = komm.ExhaustiveBitwiseDecoder(code)
    li = rng.standard_normal(code.length)
    # Posterior of each bit, in probability domain
    codewords = code.codewords()
    likelihoods = np.prod(np.exp(0.5 * li * (-1) ** codewords), axis=1)
    p1 = likelihoods @ codewords / likelihoods.sum()
    np.testing.assert_allclose(decoder.decode_to_codeword(li), np.log((1 - p1) / p1))


def test_exhaustive_bitwise_reliable_input(rng):
    code = komm.GolayCode()
    decoder = komm.ExhaustiveBitwiseDecoder(code)
    u = rng.integers(0, 2, (5, code.dimension))
    lo = decoder.decode(1000.0 * (-1) ** code.encode(u))
    assert np.all(np.isfinite(lo))
    np.testing.assert_equal(lo < 0, u == 1)
