import pytest

import numpy as np
import komm


class TestHammingCode:
    code = komm.HammingCode(3)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (7, 4, 3)

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


class TestGolayCode:
    code = komm.GolayCode()

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (23, 12, 7)

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution, [1, 0, 0, 0, 0, 0, 0, 253, 506, 0, 0, 1288, 1288, 0, 0, 506, 253, 0, 0, 0, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution, [1, 23, 253, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)


class TestExtndedGolayCode:
    code = komm.GolayCode(extended=True)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (24, 12, 8)

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution, [1, 0, 0, 0, 0, 0, 0, 0, 759, 0, 0, 0, 2576, 0, 0, 0, 759, 0, 0, 0, 0, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution, [1, 24, 276, 2024, 1771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_codewords(self):
        n, k = self.code.length, self.code.dimension
        codewords = self.code.codeword_table
        assert codewords.shape == (2**k, n)


@pytest.mark.parametrize('length,', range(2, 11))
def test_repetition_code(length):
    code1 = komm.RepetitionCode(length)
    code2 = komm.BlockCode(generator_matrix=np.ones((1, length), dtype=int))
    assert np.array_equal(code1.codeword_weight_distribution, code2.codeword_weight_distribution)
    assert np.array_equal(code1.coset_leader_weight_distribution, code2.coset_leader_weight_distribution)


@pytest.mark.parametrize('length,', range(2, 11))
def test_single_parity_check_code(length):
    code1 = komm.SingleParityCheckCode(length)
    code2 = komm.BlockCode(parity_check_matrix=np.ones((1, length), dtype=int))
    assert np.array_equal(code1.codeword_weight_distribution, code2.codeword_weight_distribution)
    assert np.array_equal(code1.coset_leader_weight_distribution, code2.coset_leader_weight_distribution)


class TestReedMuller:
    code = komm.ReedMullerCode(2, 4)

    def test_parameters(self):
        n, k, d = self.code.length, self.code.dimension, self.code.minimum_distance
        assert (n, k, d) == (16, 11, 4)

    def test_generator_matrix(self):
        assert np.array_equal(self.code.generator_matrix, [komm.int2binlist(i) for i in [0x8888, 0xa0a0, 0xaa00, 0xc0c0, 0xcc00, 0xf000, 0xaaaa, 0xcccc, 0xf0f0, 0xff00, 0xffff]])

    def test_weight_distributions(self):
        assert np.array_equal(self.code.codeword_weight_distribution, [1, 0, 0, 0, 140, 0, 448, 0, 870, 0, 448, 0, 140, 0, 0, 0, 1])
        assert np.array_equal(self.code.coset_leader_weight_distribution, [1, 16, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_GH_orthogonality(self):
        k, m = self.code.dimension, self.code.redundancy
        G = self.code.generator_matrix
        H = self.code.parity_check_matrix
        assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, m), dtype=int))

    def test_reed_partitions(self):
        # Lin.Costello.04, p. 111--113.
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
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]])

    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='zero-termination')
    assert (code.length, code.dimension, code.minimum_distance) == (8, 3, 3)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0,0,0], [0,0,1,1,0,1,0,0], [0,0,0,0,1,1,0,1]])

    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='direct-truncation')
    assert (code.length, code.dimension, code.minimum_distance) == (6, 3, 2)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0], [0,0,1,1,0,1], [0,0,0,0,1,1]])

    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='tail-biting')
    assert (code.length, code.dimension, code.minimum_distance) == (6, 3, 3)
    assert np.array_equal(code.generator_matrix, [[1,1,0,1,0,0], [0,0,1,1,0,1], [0,1,0,0,1,1]])

    # Lin.Costello.04, p. 586--587.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=6, mode='tail-biting')
    assert (code.length, code.dimension, code.minimum_distance) == (12, 6, 3)
    assert np.array_equal(code.generator_matrix[0, :], [1,1,1,0,1,1,0,0,0,0,0,0])

    # Lin.Costello.04, p. 587--590.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode='tail-biting')
    assert (code.length, code.dimension, code.minimum_distance) == (10, 5, 3)
    gen_mat = [[1,0,0,0,0,1,0,1,0,0], [0,0,1,0,0,0,0,1,0,1], [0,1,0,0,1,0,0,0,0,1], [0,1,0,1,0,0,1,0,0,0], [0,0,0,1,0,1,0,0,1,0]]
    assert np.array_equal(code.generator_matrix, gen_mat)


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials', [
    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
    ([[0o7, 0o5]], [0o7]),
])
def test_terminated_convolutional_code_zero_termination(feedforward_polynomials, feedback_polynomials):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode='zero-termination')
    for message_int in range(2**code.dimension):
        print(message_int, 2**code.dimension)
        message = komm.int2binlist(message_int, width=code.dimension)
        tail = np.dot(message, code._tail_projector) % 2
        input_sequence = komm.pack(np.concatenate([message, tail]), width=convolutional_code._num_input_bits)
        _, fs = convolutional_code._finite_state_machine.process(input_sequence, initial_state=0)
        assert fs == 0


@pytest.mark.parametrize('feedforward_polynomials',
    [[[0o7, 0o5]], [[0o3, 0o2, 0o3], [0o2, 0o1, 0o1]], [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]]])
@pytest.mark.parametrize('mode', ['zero-termination', 'direct-truncation', 'tail-biting'])
def test_terminated_convolutional_code_encoders(mode, feedforward_polynomials):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode=mode)
    for i in range(2**code.dimension):
        message = komm.int2binlist(i, width=code.dimension)
        assert np.array_equal(code.encode(message, method='generator_matrix'),
                              code.encode(message, method='finite_state_machine'))


def test_terminated_convolutional_golay():
    # Lin.Costello.04, p. 602.
    feedforward_polynomials = [[3,0,1,0,3,1,1,1], [0,3,1,1,2,3,1,0], [2,2,3,0,0,2,3,1], [0,2,0,3,2,2,2,3]]
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='tail-biting')
    assert (code.length, code.dimension, code.minimum_distance) == (24, 12, 8)


def test_terminated_convolutional_code_viterbi():
    # Lin.Costello.04, p. 522--523.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b011, 0b101, 0b111]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode='zero-termination')
    recvword = np.array([1,1,0, 1,1,0, 1,1,0, 1,1,1, 0,1,0, 1,0,1, 1,0,1])
    message_hat = code.decode(recvword, method='viterbi_hard')
    assert np.array_equal(message_hat, [1,1,0,0,1])

    # Ryan.Lin.09, p. 176--177.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=4, mode='direct-truncation')
    recvword = np.array([-0.7,-0.5, -0.8,-0.6, -1.1,+0.4, +0.9,+0.8])
    message_hat = code.decode(recvword, method='viterbi_soft')
    assert np.array_equal(message_hat, [1,0,0,0])

    # Abrantes.10, p. 307.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=10, mode='direct-truncation')
    recvword = np.array([1,1, 0,0, 0,0, 0,0, 1,0, 0,1, 0,0, 0,1, 0,1, 1,1])
    message_hat = code.decode(recvword, method='viterbi_hard')
    assert np.array_equal(message_hat, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])

    # Abrantes.10, p. 313.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode='direct-truncation')
    recvword = -np.array([-0.6,+0.8, +0.3,-0.6, +0.1,+0.1, +0.7,+0.1, +0.6,+0.4])
    message_hat = code.decode(recvword,  method='viterbi_soft')
    assert np.array_equal(message_hat, [1, 0, 1, 0, 0])


def test_terminated_convolutional_code_bcjr():
    # Abrantes.10, p. 434--437.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=4, mode='zero-termination')
    recvword = -np.array([+0.3,+0.1, -0.5,+0.2, +0.8,+0.5, -0.5,+0.3, +0.1,-0.7, +1.5,-0.4])
    message_llr = code.decode(recvword, method='bcjr', output_type='soft', SNR=1.25)
    assert np.allclose(-message_llr, [1.78, 0.24, -1.97, 5.52], atol=0.05)
    message_hat = code.decode(recvword, method='bcjr', output_type='hard', SNR=1.25)
    assert np.allclose(message_hat, [1, 1, 0, 1])

    # Lin.Costello.04, p. 572--575.
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b1]], feedback_polynomials=[0b11])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode='zero-termination')
    recvword = -np.array([+0.8,+0.1, +1.0,-0.5, -1.8,+1.1, +1.6,-1.6])
    message_llr = code.decode(recvword, method='bcjr', output_type='soft', SNR=0.25)
    assert np.allclose(-message_llr, [0.48, 0.62, -1.02], atol=0.05)
    message_hat = code.decode(recvword, method='bcjr', output_type='hard', SNR=0.25)
    assert np.allclose(message_hat, [1, 1, 0])
