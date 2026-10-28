import numpy as np

from . import RADIUS, THICK, THIN, arrow, canvas, text

STEP = 64


def lfsr(sequence):
    exponents = sequence.feedback_polynomial.exponents()
    n = exponents[-1]
    # Bit i at (STEP * i, 0); adder above
    hub = np.array([16, 80])
    xlim = (-16 - THIN / 2, STEP * n + 32)
    ylim = (-RADIUS - THICK / 2, hub[1] + RADIUS + THICK / 2)
    fig, ax = canvas(xlim, ylim, scale=(1, 1), xaxis=False, yaxis=False)
    adder(ax, hub)
    wire(ax, (0, 80), (-16, 80), (-16, 0), (16, 0))
    for k in range(1, n + 1):
        register(ax, (STEP * k - 32, 0))
    for k in range(1, n):
        arrow(ax, (STEP * k - 16, 0), (STEP * k + 16, 0))
    arrow(ax, (STEP * n - 16, 0), (STEP * n + 32, 0))
    # One lane per tap, the last on top
    for lane, i in enumerate(reversed(exponents[1:])):
        y = hub[1] - 16 * lane
        dot(ax, (STEP * i, 0))
        wire(ax, (STEP * i, 0), (STEP * i, y), (48, y), hub, shrinkB=RADIUS)
    for i in range(n + 1):
        # Digits centered, as in the nodes
        text(ax, (STEP * i, -16), f"${i}$", ((0, 0), "center", "center_baseline"))
    return fig


def register(ax, xy):
    ax.plot(*xy, "s", color="black", ms=2 * RADIUS, mfc="none", mew=THICK)


def adder(ax, xy):
    # Circled plus, as large as a node
    for marker, size in [("o", 2 * RADIUS), ("+", RADIUS)]:
        ax.plot(*xy, marker, color="black", ms=size, mfc="none", mew=THICK)


def dot(ax, xy):
    ax.plot(*xy, "o", color="black", ms=3.6, mew=0)


def wire(ax, *points, **kwargs):
    # Arrow on the last stretch only
    ax.plot(*zip(*points[:-1]), color="black", lw=THIN)
    arrow(ax, points[-2], points[-1], **kwargs)
