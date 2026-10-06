import komm

from . import markdown

max_degree = 24


def table():
    half = max_degree // 2
    rows = []
    for k in range(1, half + 1):
        row = []
        for degree in (k, k + half):
            modulus = komm.FiniteBifield(degree).modulus
            assert modulus.is_primitive()
            row += [f"${degree}$", f"`{modulus}`"]
        rows.append(row)
    header = ["Degree $k$", "Primitive polynomial $p(X)$"] * 2
    return markdown(header, rows, center=[True, False] * 2)
