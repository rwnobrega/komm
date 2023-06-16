import numpy as np

import komm


class TestGolayCode:
    code = komm.GolayCode()

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (23, 12, 7)

    def test_weight_distributions(self):
        assert np.array_equal(
            self.code.codeword_weight_distribution,
            [1, 0, 0, 0, 0, 0, 0, 253, 506, 0, 0, 1288, 1288, 0, 0, 506, 253, 0, 0, 0, 0, 0, 0, 1],
        )
        assert np.array_equal(
            self.code.coset_leader_weight_distribution,
            [1, 23, 253, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)


class TestExtendedGolayCode:
    code = komm.GolayCode(extended=True)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (24, 12, 8)

    def test_weight_distributions(self):
        assert np.array_equal(
            self.code.codeword_weight_distribution,
            [1, 0, 0, 0, 0, 0, 0, 0, 759, 0, 0, 0, 2576, 0, 0, 0, 759, 0, 0, 0, 0, 0, 0, 0, 1],
        )
        assert np.array_equal(
            self.code.coset_leader_weight_distribution,
            [1, 24, 276, 2024, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)
