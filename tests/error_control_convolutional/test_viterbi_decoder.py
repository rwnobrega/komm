import numpy as np
import pytest

import komm


def test_viterbi_stream_decoder_books():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
    traceback_length = 12
    decoder = komm.ViterbiStreamDecoder(code, traceback_length, input_type="hard")
    recvword = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    recvword_ = np.concatenate(
        [recvword, np.zeros(traceback_length * code.num_output_bits, dtype=int)]
    )
    message_hat = decoder(recvword_)
    message_hat_ = message_hat[traceback_length:]
    np.testing.assert_array_equal(message_hat_, [1, 0, 1, 1, 1, 0, 1, 1, 0, 0])


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, recvword, message_hat",
    [
        # fmt: off
        (
            [[0o7, 0o5]],
            None,
            komm.int_to_bits(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int_to_bits(0x1055CB0F07D8E51B703C77E5589DC1FCDBEC820C9A12A130C0, 200),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int_to_bits(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int_to_bits(0x1CA9300A1F7524061B0ADA89EC7E72D5906920081222BEDF0, 200),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int_to_bits(0x7577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 300),
            komm.int_to_bits(0x4B592F74786E69C9E75CFA836CFFA14F917D51AAE2C9ED60, 200),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int_to_bits(0x974B4459A5230EDE0B95CEEE67577B289B10E5F299954FCC6BCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 400),
            komm.int_to_bits(0x192F33AE3EBA2F9050B8577ADB33477613A7EA67CC7965DA40, 200),
        ),
        # fmt: on
    ],
)
def test_viterbi_stream_decoder_matlab(
    feedforward_polynomials, feedback_polynomials, recvword, message_hat
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    tblen = len(message_hat) // code.num_input_bits
    recvword = np.concatenate(
        [recvword, np.zeros(code.num_output_bits * tblen, dtype=int)]
    )
    decoder = komm.ViterbiStreamDecoder(code, traceback_length=tblen, input_type="hard")
    message_hat = np.pad(message_hat, (len(message_hat), 0), mode="constant")
    np.testing.assert_array_equal(message_hat, decoder(recvword))
