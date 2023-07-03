import numpy as np

import komm


class TestHammingCode:
    code = komm.HammingCode(3)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (7, 4, 3)

    def test_generator_matrix(self):
        G = self.code.generator_matrix
        assert np.array_equal(
            G,
            [
                [1, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
            ],
        )

    def test_parity_check_matrix(self):
        H = self.code.parity_check_matrix
        assert np.array_equal(
            H,
            [
                [0, 1, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 1],
            ],
        )

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution, [1, 0, 0, 7, 7, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution, [1, 7, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_encoding(self):
        assert np.array_equal(self.code.encode([1, 0, 0, 1]), [1, 0, 0, 1, 0, 0, 1])
        assert np.array_equal(self.code.encode([1, 0, 1, 1]), [1, 0, 1, 1, 0, 1, 0])
        assert np.array_equal(self.code.encode([1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1])

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)

    def test_decoding(self):
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 0]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 0, 1, 1, 1, 1, 0]), [1, 0, 1, 1])


class TestExtendedHammingCode:
    code = komm.HammingCode(3, extended=True)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (8, 4, 4)

    def test_generator_matrix(self):
        G = self.code.generator_matrix
        assert np.array_equal(
            G,
            [
                [1, 0, 0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 1, 0, 1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 0],
            ],
        )

    def test_parity_check_matrix(self):
        H = self.code.parity_check_matrix
        assert np.array_equal(
            H,
            [
                [1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 1],
            ],
        )

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution, [1, 0, 0, 0, 14, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution, [1, 8, 7, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_encoding(self):
        assert np.array_equal(self.code.encode([1, 0, 0, 1]), [1, 0, 0, 1, 0, 0, 1, 1])
        assert np.array_equal(self.code.encode([1, 0, 1, 1]), [1, 0, 1, 1, 0, 1, 0, 0])
        assert np.array_equal(self.code.encode([1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1, 1])

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)

    def test_decoding(self):
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 1, 0]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 1, 1, 1, 1, 1, 0, 1]), [1, 1, 1, 1])
        assert np.array_equal(self.code.decode([1, 0, 1, 1, 1, 1, 0, 0]), [1, 0, 1, 1])
