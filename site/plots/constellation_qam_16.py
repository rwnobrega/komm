import komm

from ._constellations import grid


def plot():
    const = komm.QAMConstellation(16)
    return grid(const)
