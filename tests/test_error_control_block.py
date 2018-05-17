import pytest

import numpy as np
import komm


class TestHammingCode:
    code = komm.HammingCode(3)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (7, 4, 3)

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution(), [1, 0, 0, 7, 7, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution(), [1, 7, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        n, k = self.code.length, self.code.dimension
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal((G @ H.T) % 2, np.zeros((k, n - k), dtype=np.int))

    def test_encoding(self):
        assert np.array_equal(self.code.encode([1, 0, 0, 1]), [1, 0, 0, 1, 0, 0, 1])
        assert np.array_equal(self.code.encode([1, 0, 1, 1]), [1, 0, 1, 1, 0, 1, 0])
        assert np.array_equal(self.code.encode([1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1])

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table()
        assert codewords.shape == (2**k, n)

    def test_decoding(self):
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 0]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 0, 1, 1, 1, 1, 0]), [1, 0, 1, 1])


class TestGolayCode:
    code = komm.GolayCode()

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (23, 12, 7)

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution(), [1, 0, 0, 0, 0, 0, 0, 253, 506, 0, 0, 1288, 1288, 0, 0, 506, 253, 0, 0, 0, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution(), [1, 23, 253, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        n, k = self.code.length, self.code.dimension
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal((G @ H.T) % 2, np.zeros((k, n - k), dtype=np.int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table()
        assert codewords.shape == (2**k, n)


@pytest.mark.parametrize('length, generator_polynomial, parity_check_polynomial', [
    (7, 0b1011, 0b10111),  # Hamming (7, 4)
    (23, 0b110001110101, 0b1111100100101)  # Golay (23, 12)
])
def test_cyclic_code(length, generator_polynomial, parity_check_polynomial):
    code_g = komm.CyclicCode(length, generator_polynomial=generator_polynomial)
    code_h = komm.CyclicCode(length, parity_check_polynomial=parity_check_polynomial)
    assert code_g.parity_check_polynomial == parity_check_polynomial
    assert code_h.generator_polynomial == generator_polynomial
