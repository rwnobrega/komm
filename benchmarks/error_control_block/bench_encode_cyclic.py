import numpy as np

import komm


def bench_encode_cyclic(benchmark):
    code = komm.BCHCode(8, 37)
    num_frames = 10
    u = np.random.randint(0, 2, size=(num_frames, code.dimension))
    benchmark(code.encode, u)


def bench_encode_cyclic_generator_matrix(benchmark):
    code = komm.BCHCode(8, 37)
    block_code = komm.BlockCode(generator_matrix=code.generator_matrix)
    num_frames = 10
    u = np.random.randint(0, 2, size=(num_frames, code.dimension))
    benchmark(block_code.encode, u)
