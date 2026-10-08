import komm

from . import markdown

max_degree = 24


def data():
    rows = []
    for k in range(1, max_degree + 1):
        modulus = komm.FiniteBifield(k).modulus
        assert modulus.is_primitive()
        rows.append({"k": k, "p": int(modulus)})
    return rows


def table(rows):
    cells = [[f"${row['k']}$", f"`{bin(row['p'])}`"] for row in rows]
    half = len(cells) // 2
    cells = [left + right for left, right in zip(cells[:half], cells[half:])]
    header = ["Degree $k$", "Primitive polynomial $p(X)$"] * 2
    return markdown(header, cells, center=[True, False] * 2)
