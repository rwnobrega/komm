import numpy as np

import komm


def bench_decode_convolutional_mu_2(benchmark):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o7, 0o5]],
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code,
        num_blocks=200,
        mode="zero-termination",
    )
    decoder = komm.ViterbiDecoder(code)
    num_frames = 10
    r = np.random.randint(0, 2, size=(num_frames, code.length))
    benchmark(decoder, r)


def bench_decode_convolutional_mu_6(benchmark):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o117, 0o155]],
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code,
        num_blocks=200,
        mode="zero-termination",
    )
    decoder = komm.ViterbiDecoder(code)
    num_frames = 10
    r = np.random.randint(0, 2, size=(num_frames, code.length))
    benchmark(decoder, r)


def bench_encode_convolutional_fsm(benchmark):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o7, 0o5]],
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code,
        num_blocks=1000,
        mode="zero-termination",
    )
    num_frames = 10
    u = np.random.randint(0, 2, size=(num_frames, code.dimension))
    benchmark(code.encode, u)


def bench_encode_convolutional_generator_matrix(benchmark):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o7, 0o5]],
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code,
        num_blocks=1000,
        mode="zero-termination",
    )
    block_code = komm.BlockCode(generator_matrix=code.generator_matrix)
    num_frames = 10
    u = np.random.randint(0, 2, size=(num_frames, code.dimension))
    benchmark(block_code.encode, u)
