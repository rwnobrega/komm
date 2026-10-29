import komm

from ._registers import high_rate


def plot():
    code = komm.HighRateConvolutionalCode([0o12, 0o15, 0o13, 0o11])
    return high_rate(code)
