import komm

from . import markdown

# [LC04, Tables 12.1 (a)–(c), pp. 539–540]
g_rows = [
    [0o3, 0o1],
    [0o5, 0o7],
    [0o13, 0o17],
    [0o27, 0o31],
    [0o53, 0o75],
    [0o117, 0o155],
    [0o247, 0o371],
    [0o561, 0o753],
    [0o1131, 0o1537],
    [0o2473, 0o3217],
    [0o4325, 0o6747],
    [0o10627, 0o16765],
    [0o27251, 0o37363],
    [0o1, 0o3, 0o3],
    [0o5, 0o7, 0o7],
    [0o13, 0o15, 0o17],
    [0o25, 0o33, 0o37],
    [0o47, 0o53, 0o75],
    [0o117, 0o127, 0o155],
    [0o225, 0o331, 0o367],
    [0o575, 0o623, 0o727],
    [0o1167, 0o1375, 0o1545],
    [0o2325, 0o2731, 0o3747],
    [0o5745, 0o6471, 0o7553],
    [0o2371, 0o13725, 0o14733],
    [0o1, 0o1, 0o3, 0o3],
    [0o5, 0o5, 0o7, 0o7],
    [0o13, 0o13, 0o15, 0o17],
    [0o25, 0o27, 0o33, 0o37],
    [0o45, 0o53, 0o67, 0o77],
    [0o117, 0o127, 0o155, 0o171],
    [0o257, 0o311, 0o337, 0o355],
    [0o533, 0o575, 0o647, 0o711],
    [0o1173, 0o1325, 0o1467, 0o1751],
]


def data():
    rows = []
    for g_row in g_rows:
        code = komm.LowRateConvolutionalCode(g_row)
        rows.append({
            "n": len(g_row),
            "sigma": code.degree,
            "g": g_row,
            "d_free": code.free_distance(),
        })
    return rows


def table(rows):
    cells = []
    for row in rows:
        polys = ", ".join(map(oct, row["g"]))
        cells.append([
            f"${row['n']}$",
            f"${row['sigma']}$",
            f"`[{polys}]`",
            f"${row['d_free']}$",
        ])
    header = [
        "$n$",
        "$\\sigma$",
        "$g(D) = [g_0(D) ~ \\cdots ~ g_{n-1}(D)]$",
        "$d_\\mathrm{free}$",
    ]
    return markdown(header, cells, center=[True, True, False, True])
