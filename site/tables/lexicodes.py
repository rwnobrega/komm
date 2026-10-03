import json
import os

import komm

output_file = "lexicodes.json"
max_redundancy = 29  # About 13 GiB of memory.


def dimension(n, d, previous):
    if 3 * d > 2 * n:  # Always k = 1.
        return 1
    # Redundancy grows by at most one.
    k = previous[d - 1]
    if k is None or n - k > max_redundancy:
        return None
    return komm.Lexicode(n, d).dimension


if not os.path.exists(output_file):
    dimensions = {}
    for n in range(1, 64):
        print(n)
        previous = dimensions.get(n - 1)
        dimensions[n] = [dimension(n, d, previous) for d in range(1, n + 1)]

    # One line per length.
    rows = [f'  "{n}": {json.dumps(ks)}' for n, ks in dimensions.items()]
    open(output_file, "w").write("{\n" + ",\n".join(rows) + "\n}\n")

dimensions = json.load(open(output_file, "r"))
print("| $n \\backslash d$ | " + " | ".join(f"${d}$" for d in range(1, 21)) + " |")
print("| :-: " * 21 + "|")
for n, ks in dimensions.items():
    cells = ["" if k is None else f"${k}$" for k in ks[:20]]
    cells += [""] * (20 - len(cells))
    print(f"| ${n}$ | " + " | ".join(cells) + " |")
