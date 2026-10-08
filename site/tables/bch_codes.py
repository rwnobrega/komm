import komm

from . import markdown

cached = True
max_mu = 12


def data():
    rows = []
    for mu in range(2, max_mu + 1):
        deltas = []
        for delta in list(range(2, 2 ** (mu - 1))) + [2**mu - 1]:
            print(mu, delta)
            try:
                komm.BCHCode(mu, delta)
                deltas.append(delta)
            except ValueError:
                pass
        rows.append({"mu": mu, "n": 2**mu - 1, "delta": deltas})
    return rows


def table(rows):
    cells = []
    for row in rows:
        deltas = [f"${delta}$" for delta in row["delta"]]
        cells.append([f"${row['mu']}$", f"${row['n']}$", ", ".join(deltas)])
    header = ["$\\mu$", "$n$", "Bose distances $\\delta$"]
    return markdown(header, cells, center=[True, True, False])
