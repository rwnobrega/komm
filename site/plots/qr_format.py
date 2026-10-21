import matplotlib.pyplot as plt
import numpy as np

import komm

STYLE = "default"

SIZE = 21  # Version 1
LABEL = {"ha": "center", "va": "center", "fontsize": 11}


def format_word():
    code = komm.BCHCode(mu=4, delta=7)
    mask = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    message = [0, 0, 1, 0, 1]  # Level M, mask pattern 5
    # Coefficient of X^i at position i
    return code.encode(message[::-1]) ^ mask[::-1]


def format_positions():
    # Module of X^i, as (row, column)
    first = [(r, 8) for r in [0, 1, 2, 3, 4, 5, 7, 8]]
    first += [(8, c) for c in [7, 5, 4, 3, 2, 1, 0]]
    second = [(8, SIZE - 1 - i) for i in range(8)]
    second += [(SIZE - 7 + i, 8) for i in range(7)]
    return [first, second]


def square(ax, row, col, size, color, **kwargs):
    ax.add_patch(plt.Rectangle((col, row), size, size, color=color, lw=0, **kwargs))


def function_patterns(ax):
    square(ax, 0, 0, SIZE, "0.85")  # Data area in gray
    for top, left in [(0, 0), (0, SIZE - 7), (SIZE - 7, 0)]:
        square(ax, max(top - 1, 0), max(left - 1, 0), 8, "white")  # Separator
        for k, color in enumerate(["black", "white", "black"]):
            square(ax, top + k, left + k, 7 - 2 * k, color)
    for k in range(8, SIZE - 8):  # Timing
        color = "black" if k % 2 == 0 else "white"
        square(ax, 6, k, 1, color)
        square(ax, k, 6, 1, color)
    square(ax, SIZE - 8, 8, 1, "black")  # Dark module


def plot():
    word = format_word()
    fig = plt.figure(figsize=(23 * 20 / 72, 23 * 20 / 72))
    ax = fig.add_axes((0, 0, 1, 1), xlim=(-1, SIZE + 1), ylim=(SIZE + 1, -1))
    ax.set_axis_off()
    function_patterns(ax)
    for positions in format_positions():
        for i, (r, c) in enumerate(positions):
            color, text_color = ("black", "white") if word[i] else ("white", "black")
            square(ax, r, c, 1, color)
            ax.text(c + 0.5, r + 0.5, str(i), color=text_color, **LABEL)
            ax.add_patch(
                plt.Rectangle((c, r), 1, 1, fill=False, ec="C1", lw=2, zorder=2)
            )
    return fig
