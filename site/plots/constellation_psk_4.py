import komm

from ._constellations import rings


def plot():
    const = komm.PSKConstellation(4)
    fig, _ = rings(const, ticks=[1])
    return fig
