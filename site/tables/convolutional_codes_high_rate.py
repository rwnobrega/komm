import komm

from . import markdown

# [LC04, Tables 12.1 (d) and (e), p. 540]
h_rows = [
    [0o7, 0o5, 0o3],
    [0o13, 0o15, 0o17],
    [0o27, 0o31, 0o23],
    [0o73, 0o57, 0o71],
    [0o121, 0o147, 0o123],
    [0o241, 0o227, 0o313],
    [0o477, 0o631, 0o555],
    [0o1327, 0o1423, 0o1051],
    [0o3013, 0o2137, 0o2621],
    [0o6, 0o7, 0o5, 0o1],
    [0o12, 0o15, 0o13, 0o11],
    [0o31, 0o37, 0o25, 0o33],
    [0o75, 0o57, 0o73, 0o47],
    [0o141, 0o133, 0o135, 0o107],
    [0o267, 0o315, 0o341, 0o211],
    [0o661, 0o733, 0o757, 0o535],
    [0o1371, 0o1157, 0o1723, 0o1475],
]


def table():
    rows = []
    for h_row in h_rows:
        code = komm.HighRateConvolutionalCode(h_row)
        polys = ", ".join(map(oct, h_row))
        rows.append([
            f"${len(h_row)}$",
            f"${code.degree}$",
            f"`[{polys}]`",
            f"${code.free_distance()}$",
        ])
    header = [
        "$n$",
        "$\\sigma$",
        "$h(D) = [h_0(D) ~ \\cdots ~ h_{n-1}(D)]$",
        "$d_\\mathrm{free}$",
    ]
    return markdown(header, rows, center=[True, True, False, True])
