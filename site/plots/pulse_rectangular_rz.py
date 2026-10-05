import komm

from . import label, text, xticks
from ._pulses import steps


def plot():
    pulse = komm.RectangularPulse(width=0.5)
    fig, ax = steps(pulse, [0, 0.5], (-0.5, 1.5))
    xticks(ax, [0.5, 1])
    text(ax, (0.5, 0), r"$\frac{1}{2}$", "below")
    label(ax, (1, 0), 1, "below")
    return fig
