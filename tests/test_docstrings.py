from ast import Constant, Expr, parse, walk
from pathlib import Path
from re import match

import pytest

MAX_WIDTH = 89
SRC = Path(__file__).parents[1] / "src" / "komm"


def long_example_lines(path):
    source = path.read_text()
    lines = source.splitlines()
    for node in walk(parse(source)):
        if not (isinstance(node, Expr) and isinstance(node.value, Constant)):
            continue
        in_example = False
        for i in range(node.lineno - 1, node.end_lineno or 0):
            text = lines[i].strip()
            in_example = text.startswith((">>>", "...")) or (in_example and text != "")
            if match(r"\w+Error:", text):  # Message comes from code
                continue
            if in_example and len(lines[i]) > MAX_WIDTH:
                yield i + 1


@pytest.mark.parametrize(
    "path",
    sorted(SRC.rglob("*.py")),
    ids=lambda path: path.relative_to(SRC).as_posix(),
)
def test_example_width(path):
    assert list(long_example_lines(path)) == []
