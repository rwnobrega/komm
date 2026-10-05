import importlib
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Black becomes white in dark mode
STYLE = """<style>
svg { color: black; fill: currentColor; }
@media (prefers-color-scheme: dark) { svg { color: white; } }
</style>"""


def save(fig, path):
    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=1 / 72,
        transparent=True,
        metadata={"Date": None, "Creator": None},
    )
    plt.close(fig)
    svg = buffer.getvalue()
    # Matplotlib leaves trailing spaces
    svg = re.sub(r" +\n", "\n", svg)
    # One point becomes one pixel, as in Ipe
    svg = re.sub(r'(width|height)="([\d.]+)pt"', r'\1="\2"', svg)
    svg = svg.replace("#000000", "currentColor")
    svg = re.sub(r"(<svg [^>]*>)", rf"\1\n{STYLE}", svg, count=1)
    with open(path, "w") as file:
        file.write(svg)


def main():
    site = Path(__file__).parent
    # Skip __init__.py
    for path in sorted(site.glob("plots/[!_]*.py")):
        module = importlib.import_module(f"plots.{path.stem}")
        save(module.plot(), site / "docs" / "fig" / f"{path.stem}.svg")


if __name__ == "__main__":
    main()
