import pytest

import numpy as np
import komm

from komm.util import int2binlist


def test_convolutional_simple():
     code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
     assert (code.num_output_bits, code.num_input_bits) == (2, 1)
     assert np.array_equal(code.encode([1,0,1,1,1,0,0,0]), [1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1])
     assert np.array_equal(code.constraint_lengths, [3])
     assert np.array_equal(code.memory_order, 3)
     assert np.array_equal(code.overall_constraint_length, 3)

     code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]])
     assert (code.num_output_bits, code.num_input_bits) == (3, 2)
     assert np.array_equal(code.encode([1,1,0,1,1,0,0,0]), [1,1,0,0,0,0,0,0,1,1,1,1])
     assert np.array_equal(code.constraint_lengths, [1, 1])
     assert np.array_equal(code.memory_order, 1)
     assert np.array_equal(code.overall_constraint_length, 2)


def test_convolutional_feedback():
    # Ryan.Lin.09, p. 154.
     code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
     assert np.array_equal(code.encode([1,0,0,0]), [1,1,1,0,1,1,0,0])
     code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111])
     assert np.array_equal(code.encode([1,1,1,0]), [1,1,1,0,1,1,0,0])


def test_viterbi():
     # Lin.Costello.04, p. 519-522.
     code = komm.ConvolutionalCode(feedforward_polynomials=[[0b011, 0b101, 0b111]])
     recvword_1 = np.array([1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,1,0,1])
     recvword_2 = (-1)**recvword_1
     codeword_hat_1 = code.encode(code.decode(recvword_1, method='viterbi_hard'))
     codeword_hat_2 = code.encode(code.decode(recvword_2, method='viterbi_hard'))
     assert np.array_equal(codeword_hat_1, [1,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1])
     assert np.array_equal(codeword_hat_2, [1,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1])
     #recvword = [2,3,0,2,2,1,2,2,0,2,2,2,0,3,0,3,1,2,3,0,2]


def test_fsm_forward_backward():
    # Lin.Costello.04, p. 572-575.
    def metric(z, y):
        s = (-1)**np.array(int2binlist(y, width=len(z)))
        return 0.5 * np.dot(z, s)
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [1,0]], outputs=[[0,3], [2,1]])
    z = -np.array([(0.8, 0.1), (1.0, -0.5), (-1.8, 1.1), (1.6, -1.6)])
    input_posteriors = fsm.forward_backward(z, metric_function=metric)
    llr = np.log(input_posteriors[:,0] / input_posteriors[:,1])
    assert np.allclose(-llr, [0.48, 0.62, -1.02, 2.08], atol=0.05)


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials, message, codeword', [
    ([[0o7, 0o5]], None,
     int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     int2binlist(0xbe84a1facdf49b0d258444495561c0d11f496cd12589847e89bdce6ce5555b0039b0e5589b37e56cebe5612bd2bdf7dc0000, 400)),

    ([[0o117, 0o155]], None,
     int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     int2binlist(0x3925a704c66355eb62f33de3c4512d01a6d681376ccec5f7fb8091ba4ff29b35456641cf63217ab7fd748a0560b5d4dc0000, 400)),

    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None,
     int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     int2binlist(0x6c889449f6801e93daf4e498ccf75404897d7459ce571f1581a4d05b2011986c0c8501d4000, 300)),

    ([[0o27, 0o31]], [0o27],
     int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     int2binlist(0x525114c160c91f2ac5511933f5d6ea2eceb9f48cc779f998d9d86a762d57df2a23daa7551f298d762d85d6e70e526b2c0000, 400)),
])
def test_convolutional_encoder(feedforward_polynomials, feedback_polynomials, message, codeword):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    assert np.array_equal(code.encode(message), codeword)


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials, recvword, message_hat', [
    ([[0o7, 0o5]], None,
     int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     int2binlist(0x1055cb0f07d8e51b703c77e5589dc1fcdbec820c9a12a130c0, 200)),

    ([[0o117, 0o155]], None,
     int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     int2binlist(0x1ca9300a1f7524061b0ada89ec7e72d5906920081222bedf0, 200)),

    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None,
     int2binlist(0x7577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 300),
     int2binlist(0x4b592f74786e69c9e75cfa836cffa14f917d51aae2c9ed60, 200)),

    ([[0o27, 0o31]], [0o27],
     int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     int2binlist(0x192f33ae3eba2f9050b8577adb33477613a7ea67cc7965da40, 200)),
])
def test_convolutional_decoder_viterbi(feedforward_polynomials, feedback_polynomials, recvword, message_hat):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    assert np.count_nonzero(recvword != code.encode(message_hat)) == \
           np.count_nonzero(recvword != code.encode(code.decode(recvword)))
