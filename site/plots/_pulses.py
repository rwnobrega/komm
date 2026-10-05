import numpy as np

from . import THICK, canvas, label, text, xticks, yticks


def axes(xlim, ylim, xscale):
    fig, ax = canvas(xlim, ylim, scale=(xscale, 64))
    text(ax, (xlim[1], 0), "$t$", "right")
    text(ax, (0, ylim[1]), "$p(t)$", "above")
    yticks(ax, [1])
    return fig, ax


def smooth(pulse, step):
    # Four ticks per side, 28 points apart
    fig, ax = axes((-5 * step, 5 * step), (-0.5, 1.5), 28 / step)
    t = np.linspace(-4.5 * step, 4.5 * step, 1001)
    ax.plot(t, pulse.waveform(t), color="black", lw=THICK)
    label(ax, (0, 1), 1, "above left")
    xs = step * np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    xticks(ax, xs)
    for x in xs[xs % 1 == 0]:
        # Clear of the first left sidelobe
        label(ax, (x, 0), x, "below right" if x == -step else "below")
    return fig


def steps(pulse, edges, ylim):
    fig, ax = axes((-0.5, 1.5), ylim, 64)
    edges = np.array([-0.25, *edges, 1.25])
    levels = pulse.waveform((edges[:-1] + edges[1:]) / 2)
    ax.stairs(levels, edges, baseline=None, color="black", lw=THICK)
    label(ax, (0, 1), 1, "left")
    return fig, ax
