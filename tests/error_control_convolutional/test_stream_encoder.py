import numpy as np
import pytest

import komm


def test_convolutional_stream_encoder_books():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )

    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b1101, 0b1111]],
    )
    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 0, 1, 1, 1, 0, 0, 0]),
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    )

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]],
    )
    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 1, 0, 1, 1, 0, 0, 0]),
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    )

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 0, 0, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )

    # Ibid.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
        feedback_polynomials=[0b111],
    )
    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 1, 1, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, message, codeword",
    [
        # fmt: off
        (
            [[0o7, 0o5]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0xBE84A1FACDF49B0D258444495561C0D11F496CD12589847E89BDCE6CE5555B0039B0E5589B37E56CEBE5612BD2BDF7DC0000, 400),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x3925A704C66355EB62F33DE3C4512D01A6D681376CCEC5F7FB8091BA4FF29B35456641CF63217AB7FD748A0560B5D4DC0000, 400),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x6C889449F6801E93DAF4E498CCF75404897D7459CE571F1581A4D05B2011986C0C8501D4000, 300),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x525114C160C91F2AC5511933F5D6EA2ECEB9F48CC779F998D9D86A762D57DF2A23DAA7551F298D762D85D6E70E526B2C0000, 400),
        ),
        # fmt: on
    ],
)
def test_convolutional_stream_encoder_matlab(
    feedforward_polynomials, feedback_polynomials, message, codeword
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    convolutional_encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(convolutional_encoder(message), codeword)
