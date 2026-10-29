import komm

from ._registers import low_rate


def plot():
    code = komm.LowRateConvolutionalCode([0o117, 0o155])
    return low_rate(code)
