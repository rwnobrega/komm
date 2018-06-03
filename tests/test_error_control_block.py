import pytest

import numpy as np
import komm

from komm.util import int2binlist


class TestHammingCode:
    code = komm.HammingCode(3)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (7, 4, 3)

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution(), [1, 0, 0, 7, 7, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution(), [1, 7, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=np.int))

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
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=np.int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table()
        assert codewords.shape == (2**k, n)


class TestReedMuller:
    code = komm.ReedMullerCode(2, 4)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (16, 11, 4)

    def test_generator_matrix(self):
        assert np.array_equal(self.code.generator_matrix, [int2binlist(i) for i in [0x8888, 0xa0a0, 0xaa00, 0xc0c0, 0xcc00, 0xf000, 0xaaaa, 0xcccc, 0xf0f0, 0xff00, 0xffff]])

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution(), [1, 0, 0, 0, 140, 0, 448, 0, 870, 0, 448, 0, 140, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution(), [1, 16, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=np.int))

    def test_reed_partitions(self):
        # Lin.Costello.04, p. 111-113
        reed_partitions = self.code.reed_partitions
        assert np.array_equal(reed_partitions[0], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        assert np.array_equal(reed_partitions[1], [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        assert np.array_equal(reed_partitions[8], [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]])


@pytest.mark.parametrize('length, generator_polynomial, parity_check_polynomial', [
    (7, 0b1011, 0b10111),  # Hamming (7, 4)
    (23, 0b110001110101, 0b1111100100101)  # Golay (23, 12)
])
def test_cyclic_code(length, generator_polynomial, parity_check_polynomial):
    code_g = komm.CyclicCode(length, generator_polynomial=generator_polynomial)
    code_h = komm.CyclicCode(length, parity_check_polynomial=parity_check_polynomial)
    assert code_g.parity_check_polynomial == parity_check_polynomial
    assert code_h.generator_polynomial == generator_polynomial


def test_terminated_convolutional_code():
    feedforward_polynomials = [[0b1, 0b11]]
    num_blocks = 3

    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='zero-tail')
    assert (code.length, code.dimension) == (8, 3)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0,0,0], [0,0,1,1,0,1,0,0], [0,0,0,0,1,1,0,1]])

    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='truncated')
    assert (code.length, code.dimension) == (6, 3)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0], [0,0,1,1,0,1], [0,0,0,0,1,1]])

    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='tail-biting')
    assert (code.length, code.dimension) == (6, 3)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0], [0,0,1,1,0,1], [0,1,0,0,1,1]])


@pytest.mark.parametrize('mode', ['zero-tail', 'truncated', 'tail-biting'])
def test_terminated_convolutional_code_encoders(mode):
    feedforward_polynomials = [[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]]
    num_blocks = 3
    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode=mode)
    for i in range(2**code.dimension):
        message = int2binlist(i, width=code.dimension)
        assert np.array_equal(code.encode(message, method='generator_matrix'),
                              code.encode(message, method='finite_state_machine'))


def test_terminated_convolutional_golay():
    # Lin.Costello.04, p.602
    feedforward_polynomials = [[3,0,1,0,3,1,1,1], [0,3,1,1,2,3,1,0], [2,2,3,0,0,2,3,1], [0,2,0,3,2,2,2,3]]
    num_blocks = 3
    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='tail-biting')
    assert (code.length, code.dimension, code.minimum_distance) == (24, 12, 8)


def test_terminated_convolutional_code_viterbi():
    # Lin.Costello.04, p. 522-523.
    feedforward_polynomials = [[0b011, 0b101, 0b111]]
    num_blocks = 5
    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='zero-tail')
    recvword = np.array([1,1,0, 1,1,0, 1,1,0, 1,1,1, 0,1,0, 1,0,1, 1,0,1])
    message_hat = code.decode(recvword, method='viterbi_hard')
    assert np.array_equal(message_hat, [1,1,0,0,1])

    # Ryan.Lin.09, p. 176-177
    feedforward_polynomials = [[0b111, 0b101]]
    num_blocks = 4
    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='truncated')
    recvword = np.array([-0.7,-0.5, -0.8,-0.6, -1.1,+0.4, +0.9,+0.8])
    message_hat = code.decode(recvword, method='viterbi_soft')
    assert np.array_equal(message_hat, [1,0,0,0])


def test_terminated_convolutional_code_bcjr():
    # Abrantes.10, p.434-437
    feedforward_polynomials = [[0b111, 0b101]]
    num_blocks = 4
    code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks, mode='zero-tail')
    recvword = -np.array([+0.3,+0.1, -0.5,+0.2, +0.8,+0.5, -0.5,+0.3, +0.1,-0.7, +1.5,-0.4])
    message_llr = code.decode(recvword, method='bcjr', output_type='soft', SNR=1.25)
    assert np.allclose(-message_llr, [1.78, 0.24, -1.97, 5.52], atol=0.05)
    message_hat = code.decode(recvword, method='bcjr', output_type='hard', SNR=1.25)
    assert np.allclose(message_hat, [1, 1, 0, 1])

