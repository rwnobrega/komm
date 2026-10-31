from ast import Constant, Expr, parse, walk
from itertools import takewhile
from pathlib import Path
from re import match

import pytest

MAX_WIDTH = 89
SECTIONS = ("Parameters:", "Returns:", "Raises:", "Examples:", "Notes:", "Note:")
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


def indentation(line):
    return len(line) - len(line.lstrip())


def misindented_sections(path):
    source = path.read_text()
    lines = source.splitlines()
    for node in walk(parse(source)):
        if not (isinstance(node, Expr) and isinstance(node.value, Constant)):
            continue
        docstring = lines[node.lineno - 1 : node.end_lineno]
        for i, line in enumerate(docstring):
            if not line.strip() in SECTIONS:
                continue
            title = indentation(line)
            below = [text for text in docstring[i + 1 :] if text.strip()]
            content = list(takewhile(lambda text: indentation(text) > title, below))
            # Content starts 4 spaces right of title
            if content and not min(map(indentation, content)) == title + 4:
                yield node.lineno + i


@pytest.mark.parametrize(
    "path",
    sorted(SRC.rglob("*.py")),
    ids=lambda path: path.relative_to(SRC).as_posix(),
)
def test_section_indentation(path):
    assert list(misindented_sections(path)) == []
