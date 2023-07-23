import numpy as np

import komm


class TestReedMuller:
    code = komm.ReedMullerCode(2, 4)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (16, 11, 4)

    def test_generator_matrix(self):
        assert np.array_equal(
            self.code.generator_matrix,
            [
                komm.int2binlist(i)
                for i in [0x8888, 0xA0A0, 0xAA00, 0xC0C0, 0xCC00, 0xF000, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00, 0xFFFF]
            ],
        )

    def test_weight_distributions(self):
        assert np.array_equal(
            self.code.codeword_weight_distribution, [1, 0, 0, 0, 140, 0, 448, 0, 870, 0, 448, 0, 140, 0, 0, 0, 1]
        )
        assert np.array_equal(
            self.code.coset_leader_weight_distribution, [1, 16, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_reed_partitions(self):
        # [LC04, Example 4.3]
        reed_partitions = self.code.reed_partitions
        assert np.array_equal(
            reed_partitions[0],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        )
        assert np.array_equal(
            reed_partitions[1],
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
        )
        assert np.array_equal(
            reed_partitions[8],
            [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
        )


def test_encoder():
    code = komm.ReedMullerCode(1, 5)
    encoder = komm.BlockEncoder(code)
    assert np.array_equal(
        encoder([0, 0, 0, 0, 0, 1]),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )


def test_decoder():
    code = komm.ReedMullerCode(1, 5)  # Minimum distance is 16, so it can correct up to 7 errors.
    decoder = komm.BlockDecoder(code)
    r = np.ones(32, dtype=int)
    r[[2, 10, 15, 16, 17, 19, 29]] = 0
    assert np.array_equal(decoder(r), [0, 0, 0, 0, 0, 1])
