import pytest

import numpy as np
import komm


N_bits = 2_000
awgn = komm.AWGNChannel(snr=1.0)


def _benchmark_block_decoder(benchmark, code, method, N_bits):
    N_bits = (N_bits // code.length) * code.length
    recvword = np.random.randint(2, size=N_bits)
    if getattr(code, '_decode_' + method).input_type == 'soft':
        recvword = awgn(recvword)
    benchmark(lambda: code.decode(recvword, method=method))


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['exhaustive_search_hard', 'syndrome_table', 'exhaustive_search_soft'])
@pytest.mark.parametrize('code', [
    komm.RepetitionCode(3),
    komm.RepetitionCode(5),
    komm.RepetitionCode(7),
    komm.SingleParityCheckCode(3),
    komm.SingleParityCheckCode(5),
    komm.SingleParityCheckCode(7),
    komm.HammingCode(3),
    komm.HammingCode(4),
    komm.SimplexCode(3),
    komm.SimplexCode(4),
    komm.GolayCode(),
    komm.ReedMullerCode(2, 4),
], ids=repr)
def benchmark_default_decoders(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


def rm_pairs(max_m):
    for m in range(1, max_m + 1):
        for r in range(m):
            yield (r, m)

@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['reed', 'weighted_reed'])
@pytest.mark.parametrize('code', [komm.ReedMullerCode(*rm) for rm in rm_pairs(7)], ids=repr)
def benchmark_reed_algorithm(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['meggitt'])
@pytest.mark.parametrize('code', [
    komm.CyclicCode(length=7, generator_polynomial=0o13),
    komm.CyclicCode(length=15, generator_polynomial=0o2467),
    komm.CyclicCode(length=23, generator_polynomial=0o5343),
], ids=repr)
def benchmark_meggit_decoder(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['berlekamp'])
@pytest.mark.parametrize('code', [
    komm.BCHCode(4, 1),
    komm.BCHCode(4, 2),
    komm.BCHCode(4, 3),
    komm.BCHCode(5, 1),
    komm.BCHCode(5, 2),
    komm.BCHCode(5, 3),
    komm.BCHCode(5, 5),
    komm.BCHCode(5, 7),
], ids=repr)
def benchmark_berlekamp_decoder(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['majority_logic'])
@pytest.mark.parametrize('code', [komm.RepetitionCode(n) for n in [3,13,23]], ids=repr)
def benchmark_majority_logic(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['wagner'])
@pytest.mark.parametrize('code', [komm.SingleParityCheckCode(n) for n in [3,13,23]], ids=repr)
def benchmark_wagner(benchmark, code, method, N_bits):
    _benchmark_block_decoder(benchmark, code, method, N_bits)


@pytest.mark.parametrize('N_bits', [N_bits])
@pytest.mark.parametrize('method', ['viterbi_hard', 'viterbi_soft'])
@pytest.mark.parametrize('code', [
    komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]]),
    komm.ConvolutionalCode(feedforward_polynomials=[[0o117, 0o155]]),
#    komm.ConvolutionalCode(feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]]),
], ids=repr)
def benchmark_convolutional_decoders(benchmark, code, method, N_bits):
    N_bits = (N_bits // code.num_input_bits) * code.num_output_bits
    recvword = np.random.randint(2, size=N_bits)
    if getattr(code, '_decode_' + method).input_type == 'soft':
        recvword = awgn(recvword)
    benchmark(lambda: code.decode(recvword, method=method))
