import pytest

import numpy as np
import komm


h2b = komm.util.hexstr2binarray


@pytest.mark.parametrize('generator_matrix, message, codeword', [
    ([[0o7, 0o5]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3', 200),
     h2b('00003befbd4bd486a7d736a7ecd91aa70d9c00daaaa73673bd917e2191a48b3692f88b0386aa922221a4b0d92fb35f85217d', 400)),
    ([[0o117, 0o155]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3', 200),
     h2b('0000342badf56f62dd802d6db406c04169a29ce98ff15d89311ce09380091c715495404875d3c87ccfb9285af9602f2aa853', 400)),
    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3', 200),
     h2b('0002b80a13036198804da0b2581a8f8ea739a2ebe91202aef3319272f5bc978016f92291136', 300)),
])
def test_convolutional_encoder(generator_matrix, message, codeword):
    code = komm.ConvolutionalCode(generator_matrix=generator_matrix)
    assert np.array_equal(code.encode(message), codeword)


@pytest.mark.parametrize('generator_matrix, recvword, message_hat', [
    ([[0o7, 0o5]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3d633f2a9994fa708d914deeae67773a9d07b70c4a59a22d2e9', 400),
     h2b('030c854859304137db3f83b91aa7ee3c0ed8a71be0f0d3aa08', 200)),
    ([[0o117, 0o155]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3d633f2a9994fa708d914deeae67773a9d07b70c4a59a22d2e9', 400),
     h2b('13b74c6663253ed8babcaa876cde78815b6e1f01d63f948a00', 200)),
    ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
     h2b('004934d7cc7c8efc380ffc713b2bbd47a5417faabd0e9196b3d633f2a9994fa708d914deeae', 300),
     h2b('06b79347558abe89f285ff36c15f3ae79396761e2ef49ad200', 200)),
])
def test_convolutional_decoder_viterbi(generator_matrix, recvword, message_hat):
    code = komm.ConvolutionalCode(generator_matrix=generator_matrix)
    assert np.count_nonzero(recvword != code.encode(message_hat)) == \
           np.count_nonzero(recvword != code.encode(code.decode(recvword)))
