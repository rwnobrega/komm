import komm

from ._registers import convolutional


def plot():
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o31, 0o27, 0o0], [0o0, 0o12, 0o15]],
    )
    return convolutional(code)
