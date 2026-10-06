def markdown(header, rows, center):
    # Same alignment as Prettier
    table = [header, *rows]
    widths = [max(3, *map(len, column)) for column in zip(*table)]

    def line(cells):
        return "| " + " | ".join(cells) + " |"

    def pad(text, width, center):
        left = (width - len(text)) // 2 if center else 0
        return (" " * left + text).ljust(width)

    rule = [":" + "-" * (w - 2) + ":" if c else "-" * w for w, c in zip(widths, center)]
    lines = [line(map(pad, row, widths, center)) for row in table]
    return "\n".join([lines[0], line(rule), *lines[1:]])
