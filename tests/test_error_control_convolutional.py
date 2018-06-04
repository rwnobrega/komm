import pytest

import numpy as np
import komm

from komm.util import int2binlist


def test_fsm_viterbi():
    # Sklar.01, p. 401-405.
    def metric_function(y, z):
        s = np.array(int2binlist(y, width=len(z)))
        return np.count_nonzero(z != s)
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
    z = np.array([(1, 1), (0, 1), (0, 1), (1, 0), (0, 1)])
    initial_metrics = [0.0, np.inf, np.inf, np.inf]
    input_sequences_hat, final_metrics = fsm.viterbi(z, metric_function, initial_metrics)
    assert np.allclose(final_metrics, [2.0, 2.0, 2.0, 1.0])
    assert np.array_equal(input_sequences_hat.T, [[1,1,0,0,0], [1,1,0,0,1], [1,1,1,1,0], [1,1,0,1,1]])

    # Ryan.Lin.09, p. 176-177
    def metric_function(y, z):
        y = (-1)**np.array(int2binlist(y, width=len(z)))
        return -np.dot(z, y)
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
    z = np.array([(-0.7, -0.5), (-0.8, -0.6), (-1.1, +0.4), (+0.9, +0.8)])
    initial_metrics = [0.0, np.inf, np.inf, np.inf]
    input_sequences_hat, final_metrics = fsm.viterbi(z, metric_function, initial_metrics)
    assert np.allclose(final_metrics, [-3.8, -3.4, -2.6, -2.4])
    assert np.array_equal(input_sequences_hat.T, [[1,0,0,0], [0,1,0,1], [1,1,1,0], [1,1,1,1]])


def test_fsm_forward_backward():
    # Lin.Costello.04, p. 572-575.
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [1,0]], outputs=[[0,3], [2,1]])
    input_posteriors = fsm.forward_backward(
        observed_sequence=-np.array([(0.8, 0.1), (1.0, -0.5), (-1.8, 1.1), (1.6, -1.6)]),
        metric_function=lambda y, z: 0.5 * np.dot(z, (-1)**np.array(int2binlist(y, width=len(z)))),
        initial_state_distribution=[1, 0],
        final_state_distribution=[1, 0])
    with np.errstate(divide='ignore'):
        llr = np.log(input_posteriors[:, 0] / input_posteriors[:, 1])
    assert np.allclose(-llr, [0.48, 0.62, -1.02, 2.08], atol=0.05)

    # Abrantes.10, p.434-437
    fsm = komm.FiniteStateMachine(next_states=[[0,2], [0,2], [1,3], [1,3]], outputs=[[0,3], [3,0], [1,2], [2,1]])
    input_posteriors = fsm.forward_backward(
        observed_sequence=-np.array([(0.3, 0.1), (-0.5, 0.2), (0.8, 0.5), (-0.5, 0.3), (0.1, -0.7), (1.5, -0.4)]),
        metric_function=lambda y, z: 2.5 * np.dot(z, (-1)**np.array(int2binlist(y, width=len(z)))),
        initial_state_distribution=[1, 0, 0, 0],
        final_state_distribution=[1, 0, 0, 0])
    with np.errstate(divide='ignore'):
        llr = np.log(input_posteriors[:,0] / input_posteriors[:,1])
    assert np.allclose(-llr, [1.78, 0.24, -1.97, 5.52, -np.inf, -np.inf], atol=0.05)


def test_convolutional_code():
    # Lin.Costello.04, p. 454--456
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    assert np.array_equal(code.constraint_lengths, [3])
    assert np.array_equal(code.memory_order, 3)
    assert np.array_equal(code.overall_constraint_length, 3)

    # Lin.Costello.04, p. 456--458
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]])
    assert (code.num_output_bits, code.num_input_bits) == (3, 2)
    assert np.array_equal(code.constraint_lengths, [1, 1])
    assert np.array_equal(code.memory_order, 1)
    assert np.array_equal(code.overall_constraint_length, 2)

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    assert np.array_equal(code.constraint_lengths, [2])
    assert np.array_equal(code.memory_order, 2)
    assert np.array_equal(code.overall_constraint_length, 2)

    # Ibid.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111])
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    assert np.array_equal(code.constraint_lengths, [2])
    assert np.array_equal(code.memory_order, 2)
    assert np.array_equal(code.overall_constraint_length, 2)


def test_convolutional_encoder():
    # Abrantes.10, p.307
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]), [1,1, 1,0, 0,0, 0,1, 1,0, 0,1, 0,0, 0,1, 0,1, 1,1])

    # Lin.Costello.04, p. 454--456
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 1, 1, 1, 0, 0, 0]), [1,1, 0,1, 0,0, 0,1, 0,1, 0,1, 0,0, 1,1])

    # Lin.Costello.04, p. 456--458
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]])
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder([1,1, 0,1, 1,0, 0,0]), [1,1,0, 0,0,0, 0,0,1, 1,1,1])

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 0, 0]), [1,1, 1,0, 1,1, 0,0])

    # Ibid.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111])
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 1, 1, 0]), [1,1, 1,0, 1,1, 0,0])


def test_convolutional_decoder_viterbi():
    # Abrantes.10, p.307
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    traceback_length = 12
    convolutional_decoder = komm.ConvolutionalDecoderViterbi(code, traceback_length, input_type='hard')
    recvword = np.array([1,1, 0,0, 0,0, 0,0, 1,0, 0,1, 0,0, 0,1, 0,1, 1,1])
    recvword_ = np.concatenate([recvword, np.zeros(traceback_length*code.num_output_bits, dtype=np.int)])
    message_hat = convolutional_decoder(recvword_)
    message_hat_ = message_hat[traceback_length :]
    assert np.array_equal(message_hat_, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])


def test_convolutional_decoder_bcjr():
    # Lin.Costello.04, p. 572-575.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b1]], feedback_polynomials=[0b11])
    convolutional_decoder_1 = komm.ConvolutionalDecoderBCJR(code, channel_snr=0.25, input_type='soft', output_type='soft')
    convolutional_decoder_2 = komm.ConvolutionalDecoderBCJR(code, channel_snr=0.25, input_type='soft', output_type='hard')
    recvword = -np.array([+0.8,+0.1, +1.0,-0.5, -1.8,+1.1, +1.6,-1.6])
    assert np.allclose(-convolutional_decoder_1(recvword), [0.48, 0.62, -1.02], atol=0.05)
    assert np.allclose(convolutional_decoder_2(recvword), [1, 1, 0])

    # Abrantes.10, p.434-437
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_decoder_1 = komm.ConvolutionalDecoderBCJR(code, channel_snr=1.25, input_type='soft', output_type='soft')
    convolutional_decoder_2 = komm.ConvolutionalDecoderBCJR(code, channel_snr=1.25, input_type='soft', output_type='hard')
    recvword = -np.array([+0.3,+0.1, -0.5,+0.2, +0.8,+0.5, -0.5,+0.3, +0.1,-0.7, +1.5,-0.4])
    assert np.allclose(-convolutional_decoder_1(recvword), [1.78, 0.24, -1.97, 5.52], atol=0.05)
    assert np.allclose(convolutional_decoder_2(recvword), [1, 1, 0, 1])


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
def test_convolutional_encoder_2(feedforward_polynomials, feedback_polynomials, message, codeword):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    assert np.array_equal(convolutional_encoder(message), codeword)


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
def test_convolutional_decoder_viterbi_2(feedforward_polynomials, feedback_polynomials, recvword, message_hat):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    L = len(message_hat) // code.num_input_bits
    convolutional_encoder = komm.ConvolutionalEncoder(code)
    recvword = np.concatenate([recvword, np.zeros(code.num_output_bits*L)])
    convolutional_decoder = komm.ConvolutionalDecoderViterbi(code, traceback_length=L, input_type='hard')
    message_hat = np.pad(message_hat, (len(message_hat), 0), mode='constant')
    assert np.array_equal(message_hat, convolutional_decoder(recvword))
