import numpy as np

from . import RADIUS, THICK, canvas, line, node, text

# Spacing between states, in points
COLUMN = 160
ROW = 64


def diagram(trellis):
    edge = RADIUS + THICK / 2
    xlim = (-edge, COLUMN * len(trellis) + edge)
    ylim = (-ROW * (trellis.widths.max() - 1) - edge, edge)
    fig, ax = canvas(xlim, ylim, scale=(1, 1), xaxis=False, yaxis=False)
    for t, width in enumerate(trellis.widths):
        for s in range(width):
            node(ax, (COLUMN * t, -ROW * s), f"${s}$")
    for t, section in enumerate(trellis):
        branches(ax, section, COLUMN * t)
    return fig


def branches(ax, section, left):
    states, inputs = np.nonzero(section.transitions >= 0)
    nexts = section.transitions[states, inputs]
    bottom = (section.num_states - 1, section.num_next_states - 1)
    for s, x, n in zip(states, inputs, nexts):
        start = np.array([left, -ROW * s])
        end = np.array([left + COLUMN, -ROW * n])
        length = np.hypot(*(end - start))
        d = (end - start) / length
        line(ax, start + RADIUS * d, end - RADIUS * d)
        # Label halfway to the first crossing
        ds, dn = s - states, n - nexts
        cross = ds * dn < 0
        stops = [*(ds[cross] / (ds[cross] - dn[cross]) * length), length - RADIUS]
        point = start + (RADIUS + min(stops)) / 2 * d
        # Rightward normal; below for the bottom branch
        normal = np.array([-d[1], d[0]])
        if normal[0] < 0 or ((s, n) == bottom and normal[1] > 0):
            normal = -normal
        # Slanted branches need more room
        distance = 14 if d[1] else 11
        y = section.outputs[s, x]
        text(ax, point + distance * normal, f"${x}/{y}$", "center")
