import numpy as np

import komm


def bench_decode_hamming_3_exhaustive_search_hard(benchmark):
    code = komm.HammingCode(3)
    decoder = komm.BlockDecoder(code, method="exhaustive_search_hard")
    n_blocks = 1000
    r = np.random.randint(0, 2, size=n_blocks * code.length)
    benchmark(decoder, r)


def bench_decode_hamming_3_syndrome_table(benchmark):
    code = komm.HammingCode(3)
    decoder = komm.BlockDecoder(code, method="syndrome_table")
    n_blocks = 1000
    r = np.random.randint(0, 2, size=n_blocks * code.length)
    benchmark(decoder, r)
