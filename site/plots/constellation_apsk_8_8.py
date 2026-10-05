import komm

from ._constellations import rings


def plot():
    const = komm.APSKConstellation(
        orders=(8, 8),
        amplitudes=(1.0, 2.0),
        phase_offsets=(0.0, 1 / 16),
    )
    fig, _ = rings(const, ticks=[1, 2])
    return fig
