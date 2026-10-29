import komm

from ._registers import feedback


def plot():
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0o27, 0o31]],
        feedback_polynomials=[0o27],
    )
    return feedback(code)
