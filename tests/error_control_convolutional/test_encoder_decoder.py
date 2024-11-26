import numpy as np
import pytest

import komm


def test_convolutional_stream_encoder_books():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        convolutional_encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )

    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b1101, 0b1111]],
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        convolutional_encoder([1, 0, 1, 1, 1, 0, 0, 0]),
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    )

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]],
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        convolutional_encoder([1, 1, 0, 1, 1, 0, 0, 0]),
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    )

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        convolutional_encoder([1, 0, 0, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )

    # Ibid.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
        feedback_polynomials=[0b111],
    )
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        convolutional_encoder([1, 1, 1, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )


def test_convolutional_stream_decoder_books():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    traceback_length = 12
    convolutional_decoder = komm.ConvolutionalStreamDecoder(
        code, traceback_length, input_type="hard"
    )
    recvword = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    recvword_ = np.concatenate(
        [recvword, np.zeros(traceback_length * code.num_output_bits, dtype=int)]
    )
    message_hat = convolutional_decoder(recvword_)
    message_hat_ = message_hat[traceback_length:]
    np.testing.assert_array_equal(message_hat_, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, message, codeword",
    [
        # fmt: off
        (
            [[0o7, 0o5]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(0xBE84A1FACDF49B0D258444495561C0D11F496CD12589847E89BDCE6CE5555B0039B0E5589B37E56CEBE5612BD2BDF7DC0000, 400),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(0x3925A704C66355EB62F33DE3C4512D01A6D681376CCEC5F7FB8091BA4FF29B35456641CF63217AB7FD748A0560B5D4DC0000, 400),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(0x6C889449F6801E93DAF4E498CCF75404897D7459CE571F1581A4D05B2011986C0C8501D4000, 300),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int2binlist(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int2binlist(0x525114C160C91F2AC5511933F5D6EA2ECEB9F48CC779F998D9D86A762D57DF2A23DAA7551F298D762D85D6E70E526B2C0000, 400),
        ),
        # fmt: on
    ],
)
def test_convolutional_stream_encoder_matlab(
    feedforward_polynomials,
    feedback_polynomials,
    message,
    codeword,
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(convolutional_encoder(message), codeword)


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, recvword, message_hat",
    [
        # fmt: off
        (
            [[0o7, 0o5]],
            None,
            komm.int2binlist(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int2binlist(0x1055CB0F07D8E51B703C77E5589DC1FCDBEC820C9A12A130C0, 200),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int2binlist(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int2binlist(0x1CA9300A1F7524061B0ADA89EC7E72D5906920081222BEDF0, 200),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int2binlist(0x7577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 300),
            komm.int2binlist(0x4B592F74786E69C9E75CFA836CFFA14F917D51AAE2C9ED60, 200),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int2binlist(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int2binlist(0x192F33AE3EBA2F9050B8577ADB33477613A7EA67CC7965DA40, 200),
        ),
        # fmt: on
    ],
)
def test_convolutional_stream_decoder_matlab(
    feedforward_polynomials,
    feedback_polynomials,
    recvword,
    message_hat,
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    tblen = len(message_hat) // code.num_input_bits
    recvword = np.concatenate([recvword, np.zeros(code.num_output_bits * tblen)])
    convolutional_decoder = komm.ConvolutionalStreamDecoder(
        code, traceback_length=tblen, input_type="hard"
    )
    message_hat = np.pad(message_hat, (len(message_hat), 0), mode="constant")
    np.testing.assert_array_equal(message_hat, convolutional_decoder(recvword))
