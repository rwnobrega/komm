import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

from . import GAP, HEAD, RADIUS, THICK, THIN, arrow, canvas, node, text, toward

BEND = 0.3125


def mealy(machine, positions):
    nodes = [f"${s}$" for s in range(machine.num_states)]
    edges = [
        (s, n, f"${x}/{machine.outputs[s, x]}$")
        for (s, x), n in np.ndenumerate(machine.transitions)
    ]
    return diagram(positions, nodes, edges)


def moore(machine, positions):
    nodes = [f"${s}/{y}$" for s, y in enumerate(machine.outputs)]
    edges = [(s, n, f"${x}$") for (s, x), n in np.ndenumerate(machine.transitions)]
    return diagram(positions, nodes, edges)


def diagram(positions, nodes, edges):
    points = np.array(positions, dtype=float)
    center = points.mean(axis=0)
    # Loops and labels may spill out
    lo = points.min(axis=0) - RADIUS - THICK / 2
    hi = points.max(axis=0) + RADIUS + THICK / 2
    xlim, ylim = (lo[0], hi[0]), (lo[1], hi[1])
    fig, ax = canvas(xlim, ylim, scale=(1, 1), xaxis=False, yaxis=False)
    for point, string in zip(points, nodes):
        node(ax, point, string)
    pairs = {(s, n) for s, n, _ in edges}
    for s, n, string in edges:
        a, b = points[s], points[n]
        if s == n:
            loop(ax, a, unit(a - center), string)
        elif (n, s) in pairs:
            bend(ax, a, b, string)
        else:
            straight(ax, a, b, string, center)
    return fig


def unit(v):
    return v / np.hypot(*v)


def straight(ax, a, b, string, center):
    arrow(ax, a, b, shrinkA=RADIUS, shrinkB=RADIUS)
    mid = (a + b) / 2
    normal = unit(np.array([a[1] - b[1], b[0] - a[0]]))
    # Label away from the center
    if np.dot(normal, mid - center) < 0:
        normal = -normal
    # Closer to edges than to points
    text(ax, mid, string, toward(normal, GAP / 2))


def bend(ax, a, b, string):
    # Bend right, away from the reverse
    style = f"arc3,rad={BEND}"
    arrow(ax, a, b, connectionstyle=style, shrinkA=RADIUS, shrinkB=RADIUS)
    side = np.array([b[1] - a[1], a[0] - b[0]])
    text(ax, (a + b) / 2 + BEND / 2 * side, string, toward(unit(side)))


def loop(ax, a, u, string):
    # Out and back along u, in radii
    c = np.sqrt(0.5)
    uv = [[c, -c], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [c, c]]
    vertices = a + RADIUS * np.array(uv) @ [u, [-u[1], u[0]]]
    path = Path(vertices, [Path.MOVETO] + [Path.CURVE4] * 6)
    props = dict(arrowstyle=HEAD, mutation_scale=1, lw=THIN, color="black")
    ax.add_patch(FancyArrowPatch(path=path, clip_on=False, **props))
    text(ax, a + 3 * RADIUS * u, string, toward(u))
