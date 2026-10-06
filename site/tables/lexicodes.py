import json
from pathlib import Path

import komm

from . import markdown

output_file = Path(__file__).with_suffix(".json")
max_redundancy = 29  # About 13 GiB of memory.


def dimension(n, d, previous):
    if 3 * d > 2 * n:  # Always k = 1.
        return 1
    # Redundancy grows by at most one.
    k = previous[d - 1]
    if k is None or n - k > max_redundancy:
        return None
    return komm.Lexicode(n, d).dimension


def compute():
    dimensions = {}
    for n in range(1, 64):
        print(n)
        previous = dimensions.get(n - 1)
        dimensions[n] = [dimension(n, d, previous) for d in range(1, n + 1)]

    # One line per length.
    rows = [f'  "{n}": {json.dumps(ks)}' for n, ks in dimensions.items()]
    output_file.write_text("{\n" + ",\n".join(rows) + "\n}\n")


def table():
    if not output_file.exists():
        compute()
    dimensions = json.loads(output_file.read_text())
    rows = []
    for n, ks in dimensions.items():
        cells = ["" if k is None else f"${k}$" for k in ks[:20]]
        cells += [""] * (20 - len(cells))
        rows.append([f"${n}$", *cells])
    header = ["$n \\backslash d$", *(f"${d}$" for d in range(1, 21))]
    return markdown(header, rows, center=[True] * 21)
