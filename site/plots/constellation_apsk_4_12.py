import numpy as np

import komm

from ._constellations import rings


def plot():
    const = komm.APSKConstellation(
        orders=(4, 12),
        amplitudes=(np.sqrt(2), 3.0),
        phase_offsets=(1 / 8, 0.0),
    )
    fig, _ = rings(const, ticks=[1, 2, 3])
    return fig
