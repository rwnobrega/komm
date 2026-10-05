import komm

from ._constellations import grid


def plot():
    const = komm.CrossQAMConstellation(32)
    return grid(const)
