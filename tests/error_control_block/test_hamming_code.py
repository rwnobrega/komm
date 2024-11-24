import numpy as np
import pytest

import komm


class TestHammingCode:
    code = komm.HammingCode(3)

    def test_parameters(self):
        n, k, m = self.code.length, self.code.dimension, self.code.redundancy
        assert (n, k, m) == (7, 4, 3)
        assert self.code.minimum_distance() == 3

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
        H = self.code.check_matrix
        assert np.array_equal(
            H,
            [
                [1, 1, 0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 0, 1],
            ],
        )

    def test_weight_distributions(self):
        assert np.array_equal(
            self.code.codeword_weight_distribution(), [1, 0, 0, 7, 7, 0, 0, 1]
        )
        assert np.array_equal(
            self.code.coset_leader_weight_distribution(), [1, 7, 0, 0, 0, 0, 0, 0]
        )

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    @pytest.mark.parametrize(
        "u, v",
        [
            ([1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]),
            ([1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0]),
            ([1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
        ],
    )
    def test_enc_mapping(self, u, v):
        assert np.array_equal(self.code.enc_mapping(u), v)
        assert np.array_equal(self.code.inv_enc_mapping(v), u)

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codewords()
        assert codewords.shape == (2**k, n)

    def test_encoder(self):
        encoder = komm.BlockEncoder(self.code)
        assert np.array_equal(
            encoder([1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        )

    def test_decoder(self):
        decoder = komm.BlockDecoder(self.code)
        assert np.array_equal(
            decoder([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0]),
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        )


class TestExtendedHammingCode:
    code = komm.HammingCode(3, extended=True)

    def test_parameters(self):
        n, k, m = self.code.length, self.code.dimension, self.code.redundancy
        assert (n, k, m) == (8, 4, 4)
        assert self.code.minimum_distance() == 4

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
        H = self.code.check_matrix
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
        assert np.array_equal(
            self.code.codeword_weight_distribution(), [1, 0, 0, 0, 14, 0, 0, 0, 1]
        )
        assert np.array_equal(
            self.code.coset_leader_weight_distribution(), [1, 8, 7, 0, 0, 0, 0, 0, 0]
        )

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    @pytest.mark.parametrize(
        "u, v",
        [
            ([1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1, 1]),
            ([1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0]),
            ([1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
        ],
    )
    def test_enc_mapping(self, u, v):
        assert np.array_equal(self.code.enc_mapping(u), v)
        assert np.array_equal(self.code.inv_enc_mapping(v), u)

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codewords()
        assert codewords.shape == (2**k, n)

    def test_encoder(self):
        encoder = komm.BlockEncoder(self.code)
        assert np.array_equal(
            encoder([1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        )

    def test_decoder(self):
        decoder = komm.BlockDecoder(self.code)
        assert np.array_equal(
            decoder(
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0]
            ),
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        )
