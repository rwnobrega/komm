import json
import os

import komm

output_file = "bch_codes.json"

if not os.path.exists(output_file):
    bose = {}
    for mu in range(2, 11):
        bose[mu] = []
        for delta in list(range(2, 2 ** (mu - 1))) + [2**mu - 1]:
            print(mu, delta)
            try:
                code = komm.BCHCode(mu, delta)
                bose[mu].append(delta)
            except ValueError:
                pass

    json.dump(bose, open("bch_codes.json", "w"), indent=4)

bose = json.load(open("bch_codes.json", "r"))
print("| $\\mu$ | $n$ |  Bose distances $\\delta$ |")
print("| :-: | :-: | --- |")
for mu in bose:
    n = 2 ** int(mu) - 1
    deltas = [f"${delta}$" for delta in bose[mu]]
    print(f"| ${mu}$ | ${n}$ | {', '.join(deltas)} |")
