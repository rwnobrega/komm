import numpy as np
import pytest

import komm


def test_convolutional_code():
    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    assert np.array_equal(code.constraint_lengths, [3])
    assert np.array_equal(code.memory_order, 3)
    assert np.array_equal(code.overall_constraint_length, 3)

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]]
    )
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
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111]
    )
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


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials",
    [
        ([[0o7, 0o5]], None),
        ([[0o117, 0o155]], None),
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o27, 0o31]], [0o27]),
    ],
)
def test_convolutional_space_state_representation_2(
    feedforward_polynomials, feedback_polynomials
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    n, k, nu = code.num_output_bits, code.num_input_bits, code.overall_constraint_length

    A = code.state_matrix
    B = code.control_matrix
    C = code.observation_matrix
    D = code.transition_matrix

    input_bits = np.random.randint(2, size=100 * k)
    output_bits = np.empty(n * input_bits.size // k, dtype=int)

    s = np.zeros(nu, dtype=int)

    for t, u in enumerate(np.reshape(input_bits, shape=(-1, k))):
        s, v = (np.dot(s, A) + np.dot(u, B)) % 2, (np.dot(s, C) + np.dot(u, D)) % 2
        output_bits[t * n : (t + 1) * n] = v

    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(output_bits, convolutional_encoder(input_bits))


def test_convolutional_stream_encoder():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(
        convolutional_encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )

    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1101, 0b1111]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(
        convolutional_encoder([1, 0, 1, 1, 1, 0, 0, 0]),
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    )

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]]
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(
        convolutional_encoder([1, 1, 0, 1, 1, 0, 0, 0]),
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    )

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 0, 0, 0]), [1, 1, 1, 0, 1, 1, 0, 0])

    # Ibid.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111]
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder([1, 1, 1, 0]), [1, 1, 1, 0, 1, 1, 0, 0])


def test_convolutional_stream_decoder():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    traceback_length = 12
    convolutional_decoder = komm.ConvolutionalStreamDecoder(
        code, traceback_length, input_type="hard"
    )
    recvword = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    recvword_ = np.concatenate(
        [recvword, np.zeros(traceback_length * code.num_output_bits, dtype=int)]
    )
    message_hat = convolutional_decoder(recvword_)
    message_hat_ = message_hat[traceback_length:]
    assert np.array_equal(message_hat_, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, message, codeword",
    [
        (
            [[0o7, 0o5]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(
                0xBE84A1FACDF49B0D258444495561C0D11F496CD12589847E89BDCE6CE5555B0039B0E5589B37E56CEBE5612BD2BDF7DC0000,
                400,
            ),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(
                0x3925A704C66355EB62F33DE3C4512D01A6D681376CCEC5F7FB8091BA4FF29B35456641CF63217AB7FD748A0560B5D4DC0000,
                400,
            ),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(
                0x6C889449F6801E93DAF4E498CCF75404897D7459CE571F1581A4D05B2011986C0C8501D4000,
                300,
            ),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(
                0x525114C160C91F2AC5511933F5D6EA2ECEB9F48CC779F998D9D86A762D57DF2A23DAA7551F298D762D85D6E70E526B2C0000,
                400,
            ),
        ),
    ],
)
def test_convolutional_stream_encoder_2(
    feedforward_polynomials, feedback_polynomials, message, codeword
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    assert np.array_equal(convolutional_encoder(message), codeword)


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, recvword, message_hat",
    [
        (
            [[0o7, 0o5]],
            None,
            komm.int2binlist(
                0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200,
                400,
            ),
            komm.int2binlist(0x1055CB0F07D8E51B703C77E5589DC1FCDBEC820C9A12A130C0, 200),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int2binlist(
                0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200,
                400,
            ),
            komm.int2binlist(0x1CA9300A1F7524061B0ADA89EC7E72D5906920081222BEDF0, 200),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int2binlist(
                0x7577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200,
                300,
            ),
            komm.int2binlist(0x4B592F74786E69C9E75CFA836CFFA14F917D51AAE2C9ED60, 200),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int2binlist(
                0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200,
                400,
            ),
            komm.int2binlist(0x192F33AE3EBA2F9050B8577ADB33477613A7EA67CC7965DA40, 200),
        ),
    ],
)
def test_convolutional_stream_decoder_2(
    feedforward_polynomials, feedback_polynomials, recvword, message_hat
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    L = len(message_hat) // code.num_input_bits
    recvword = np.concatenate([recvword, np.zeros(code.num_output_bits * L)])
    convolutional_decoder = komm.ConvolutionalStreamDecoder(
        code, traceback_length=L, input_type="hard"
    )
    message_hat = np.pad(message_hat, (len(message_hat), 0), mode="constant")
    assert np.array_equal(message_hat, convolutional_decoder(recvword))
