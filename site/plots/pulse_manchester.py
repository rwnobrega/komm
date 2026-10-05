import komm

from . import DASHES, label, line, text, xticks
from ._pulses import steps


def plot():
    pulse = komm.ManchesterPulse()
    fig, ax = steps(pulse, [0, 0.5, 1], (-1.5, 1.5))
    line(ax, (0, 1), (0.5, 1), dashes=DASHES)
    xticks(ax, [0.5, 1])
    # Clear of the jump at 1/2
    text(ax, (0.5, 0), r"$\frac{1}{2}$", "below right")
    label(ax, (1, 0), 1, "below")
    label(ax, (0, -1), -1, "left")
    return fig
