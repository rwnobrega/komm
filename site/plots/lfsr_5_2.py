import komm

from ._registers import lfsr


def plot():
    sequence = komm.LFSRSequence(feedback_polynomial=0b100101)
    return lfsr(sequence)
