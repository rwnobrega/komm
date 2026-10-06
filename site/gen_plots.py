import importlib
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt

SVG = {"svg.fonttype": "path", "svg.hashsalt": "komm"}

# Black becomes white in dark mode
CSS = """<style>
svg { color: black; fill: currentColor; }
@media (prefers-color-scheme: dark) { svg { color: white; } }
</style>"""


def save(fig, path, themed):
    buffer = io.StringIO()
    tight = dict(bbox_inches="tight", pad_inches=1 / 72, transparent=True)
    with plt.rc_context(SVG):
        fig.savefig(
            buffer,
            format="svg",
            metadata={"Date": None, "Creator": None},
            **(tight if themed else {}),
        )
    plt.close(fig)
    svg = buffer.getvalue()
    # Matplotlib leaves trailing spaces
    svg = re.sub(r" +\n", "\n", svg)
    # One point becomes one pixel, as in Ipe
    svg = re.sub(r'(width|height)="([\d.]+)pt"', r'\1="\2"', svg)
    if themed:
        svg = svg.replace("#000000", "currentColor")
        svg = re.sub(r"(<svg [^>]*>)", rf"\1\n{CSS}", svg, count=1)
    with open(path, "w") as file:
        file.write(svg)


def main():
    site = Path(__file__).parent
    # Skip helpers, like __init__.py
    for path in sorted(site.glob("plots/[!_]*.py")):
        module = importlib.import_module(f"plots.{path.stem}")
        # Plots with their own style stay unthemed
        style = getattr(module, "STYLE", {})
        with plt.style.context(style):
            fig = module.plot()
            save(fig, site / "docs" / "fig" / f"{path.stem}.svg", themed=not style)
        print(f"Generated {path.stem}.svg")


if __name__ == "__main__":
    main()
