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

    def test_exhaustive_search_hard_decoder(self):
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 0]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 0, 1, 1, 1, 1, 0]), [1, 0, 1, 1])
