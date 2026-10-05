import komm

from . import label, xticks
from ._pulses import steps


def plot():
    pulse = komm.RectangularPulse(width=1.0)
    fig, ax = steps(pulse, [0, 1], (-0.5, 1.5))
    xticks(ax, [1])
    label(ax, (1, 0), 1, "below")
    return fig
