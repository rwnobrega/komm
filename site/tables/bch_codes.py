import json
import os

import komm

output_file = "bch_codes.json"
max_mu = 12

if not os.path.exists(output_file):
    bose = {}
    for mu in range(2, max_mu + 1):
        bose[mu] = []
        for delta in list(range(2, 2 ** (mu - 1))) + [2**mu - 1]:
            print(mu, delta)
            try:
                code = komm.BCHCode(mu, delta)
                bose[mu].append(delta)
            except ValueError:
                pass

    # One line per mu.
    rows = [f'  "{mu}": {json.dumps(deltas)}' for mu, deltas in bose.items()]
    open(output_file, "w").write("{\n" + ",\n".join(rows) + "\n}\n")

bose = json.load(open(output_file, "r"))
print("| $\\mu$ | $n$ |  Bose distances $\\delta$ |")
print("| :-: | :-: | --- |")
for mu in bose:
    n = 2 ** int(mu) - 1
    deltas = [f"${delta}$" for delta in bose[mu]]
    print(f"| ${mu}$ | ${n}$ | {', '.join(deltas)} |")
