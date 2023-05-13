import pytest

import numpy as np
import komm


def test_convolutional_code():
    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    assert np.array_equal(code.constraint_lengths, [3])
    assert np.array_equal(code.memory_order, 3)
    assert np.array_equal(code.overall_constraint_length, 3)

    # Lin.Costello.04, p. 456--458.
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


def test_convolutional_space_state_representation():
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
    A = code.state_matrix
    B = code.control_matrix
    C = code.observation_matrix
    D = code.transition_matrix
    assert np.array_equal(A, [[0, 1], [0, 0]])
    assert np.array_equal(B, [[1, 0]])
    assert np.array_equal(C, [[1, 0], [1, 1]])
    assert np.array_equal(D, [[1, 1]])

    # Heide Gluesing-Luerssen: On the Weight Distribution of Convolutional Codes, p. 9.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1111, 0b1101]])
    A = code.state_matrix
    B = code.control_matrix
    C = code.observation_matrix
    D = code.transition_matrix
    assert np.array_equal(A, [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert np.array_equal(B, [[1, 0, 0]])
    assert np.array_equal(C, [[1, 0], [1, 1], [1, 1]])
    assert np.array_equal(D, [[1, 1]])


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials', [
    ([[0o7, 0o5]], None),
    ([[0o117, 0o155]], None),
    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
    ([[0o27, 0o31]], [0o27]),
])
def test_convolutional_space_state_representation_2(feedforward_polynomials, feedback_polynomials):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    n, k, nu = code.num_output_bits, code.num_input_bits, code.overall_constraint_length

    A = code.state_matrix
    B = code.control_matrix
    C = code.observation_matrix
    D = code.transition_matrix

    input_bits = np.random.randint(2, size=100*k)
    output_bits = np.empty(n * input_bits.size // k, dtype=int)

    s = np.zeros(nu, dtype=int)

    for t, u in enumerate(np.reshape(input_bits, newshape=(-1, k))):
        s, v = (np.dot(s, A) + np.dot(u, B)) % 2, (np.dot(s, C) + np.dot(u, D)) % 2
        output_bits[t*n : (t+1)*n] = v

    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(output_bits, convolutional_encoder(input_bits))


def test_convolutional_stream_encoder():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]), [1,1, 1,0, 0,0, 0,1, 1,0, 0,1, 0,0, 0,1, 0,1, 1,1])

    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 1, 1, 1, 0, 0, 0]), [1,1, 0,1, 0,0, 0,1, 0,1, 0,1, 0,0, 1,1])

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1,1, 0,1, 1,0, 0,0]), [1,1,0, 0,0,0, 0,0,1, 1,1,1])

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 0, 0]), [1,1, 1,0, 1,1, 0,0])

    # Ibid.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 1, 1, 0]), [1,1, 1,0, 1,1, 0,0])


def test_convolutional_stream_decoder():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    traceback_length = 12
    convolutional_decoder = komm.ConvolutionalStreamDecoder(code, traceback_length, input_type='hard')
    recvword = np.array([1,1, 0,0, 0,0, 0,0, 1,0, 0,1, 0,0, 0,1, 0,1, 1,1])
    recvword_ = np.concatenate([recvword, np.zeros(traceback_length*code.num_output_bits, dtype=int)])
    message_hat = convolutional_decoder(recvword_)
    message_hat_ = message_hat[traceback_length :]
    assert np.array_equal(message_hat_, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials, message, codeword', [
    ([[0o7, 0o5]], None,
     komm.int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     komm.int2binlist(0xbe84a1facdf49b0d258444495561c0d11f496cd12589847e89bdce6ce5555b0039b0e5589b37e56cebe5612bd2bdf7dc0000, 400)),

    ([[0o117, 0o155]], None,
     komm.int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     komm.int2binlist(0x3925a704c66355eb62f33de3c4512d01a6d681376ccec5f7fb8091ba4ff29b35456641cf63217ab7fd748a0560b5d4dc0000, 400)),

    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None,
     komm.int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     komm.int2binlist(0x6c889449f6801e93daf4e498ccf75404897d7459ce571f1581a4d05b2011986c0c8501d4000, 300)),

    ([[0o27, 0o31]], [0o27],
     komm.int2binlist(0xcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 200),
     komm.int2binlist(0x525114c160c91f2ac5511933f5d6ea2eceb9f48cc779f998d9d86a762d57df2a23daa7551f298d762d85d6e70e526b2c0000, 400)),
])
def test_convolutional_stream_encoder_2(feedforward_polynomials, feedback_polynomials, message, codeword):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder(message), codeword)


@pytest.mark.parametrize('feedforward_polynomials, feedback_polynomials, recvword, message_hat', [
    ([[0o7, 0o5]], None,
     komm.int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     komm.int2binlist(0x1055cb0f07d8e51b703c77e5589dc1fcdbec820c9a12a130c0, 200)),

    ([[0o117, 0o155]], None,
     komm.int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     komm.int2binlist(0x1ca9300a1f7524061b0ada89ec7e72d5906920081222bedf0, 200)),

    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None,
     komm.int2binlist(0x7577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 300),
     komm.int2binlist(0x4b592f74786e69c9e75cfa836cffa14f917d51aae2c9ed60, 200)),

    ([[0o27, 0o31]], [0o27],
     komm.int2binlist(0x974b4459a5230ede0b95ceee67577b289b10e5f299954fcc6bcd698970bd55fe82a5e2bdd4dc8e3ff01c3f713e33eb2c9200, 400),
     komm.int2binlist(0x192f33ae3eba2f9050b8577adb33477613a7ea67cc7965da40, 200)),
])
def test_convolutional_stream_decoder_2(feedforward_polynomials, feedback_polynomials, recvword, message_hat):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    L = len(message_hat) // code.num_input_bits
    recvword = np.concatenate([recvword, np.zeros(code.num_output_bits*L)])
    convolutional_decoder = komm.ConvolutionalStreamDecoder(code, traceback_length=L, input_type='hard')
    message_hat = np.pad(message_hat, (len(message_hat), 0), mode='constant')
    assert np.array_equal(message_hat, convolutional_decoder(recvword))
