import komm

from . import markdown

cached = True
max_redundancy = 29  # About 13 GiB of memory.


def dimension(n, d, previous):
    if 3 * d > 2 * n:  # Always k = 1.
        return 1
    # Redundancy grows by at most one.
    k = previous[d - 1]
    if k is None or n - k > max_redundancy:
        return None
    return komm.Lexicode(n, d).dimension


def data():
    rows = []
    for n in range(1, 64):
        print(n)
        previous = rows[-1]["k"] if rows else None
        ks = [dimension(n, d, previous) for d in range(1, n + 1)]
        rows.append({"n": n, "k": ks})
    return rows


def table(rows):
    cells = []
    for row in rows:
        ks = ["" if k is None else f"${k}$" for k in row["k"][:20]]
        ks += [""] * (20 - len(ks))
        cells.append([f"${row['n']}$", *ks])
    header = ["$n \\backslash d$", *(f"${d}$" for d in range(1, 21))]
    return markdown(header, cells, center=[True] * 21)
