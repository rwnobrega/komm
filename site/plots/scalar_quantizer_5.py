from itertools import pairwise

import numpy as np

import komm

from . import DASHES, THICK, canvas, label, line


def plot():
    quantizer = komm.ScalarQuantizer(
        levels=[-2.0, -1.0, 0.0, 1.0, 2.0],
        thresholds=[-1.5, -0.3, 0.8, 1.4],
    )
    fig, ax = canvas((-3.5, 3.5), (-3, 3))
    edges = [-3, *quantizer.thresholds, 3]
    ax.stairs(quantizer.levels, edges, baseline=None, color="black", lw=THICK)
    # Labels stay away from the curve
    for x, (lo, hi) in zip(quantizer.thresholds, pairwise(quantizer.levels)):
        line(ax, (x, 0), (x, np.clip(0, lo, hi)), dashes=DASHES)
        line(ax, (x, -0.125), (x, 0.125))
        label(ax, (x, 0), x, "above" if x < 0 else "below")
    for y, (lo, hi) in zip(quantizer.levels, pairwise(edges)):
        if y != 0:
            line(ax, (0, y), (np.clip(0, lo, hi), y), dashes=DASHES)
            line(ax, (-0.125, y), (0.125, y))
            label(ax, (0, y), y, "left" if y > 0 else "right")
    return fig
