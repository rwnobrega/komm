import komm

from ._constellations import grid


def plot():
    const = komm.QAMConstellation(orders=(4, 2), deltas=(2.0, 4.0))
    return grid(const)
