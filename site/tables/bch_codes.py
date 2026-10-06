import json
from pathlib import Path

import komm

from . import markdown

output_file = Path(__file__).with_suffix(".json")
max_mu = 12


def compute():
    bose = {}
    for mu in range(2, max_mu + 1):
        bose[mu] = []
        for delta in list(range(2, 2 ** (mu - 1))) + [2**mu - 1]:
            print(mu, delta)
            try:
                komm.BCHCode(mu, delta)
                bose[mu].append(delta)
            except ValueError:
                pass

    # One line per mu.
    rows = [f'  "{mu}": {json.dumps(deltas)}' for mu, deltas in bose.items()]
    output_file.write_text("{\n" + ",\n".join(rows) + "\n}\n")


def table():
    if not output_file.exists():
        compute()
    bose = json.loads(output_file.read_text())
    rows = []
    for mu in bose:
        n = 2 ** int(mu) - 1
        deltas = [f"${delta}$" for delta in bose[mu]]
        rows.append([f"${mu}$", f"${n}$", ", ".join(deltas)])
    header = ["$\\mu$", "$n$", "Bose distances $\\delta$"]
    return markdown(header, rows, center=[True, True, False])
