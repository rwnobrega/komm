import importlib
import re
from pathlib import Path

# Generated text sits between these markers
MARKERS = re.compile(r"(<!-- table: (\w+) -->\n\n).*?(\n\n<!-- end table -->)", re.S)


def fill(match):
    module = importlib.import_module(f"tables.{match[2]}")
    table = module.table()
    print(f"Generated {match[2]}")
    return match[1] + table + match[3]


def main():
    site = Path(__file__).parent
    for path in sorted(site.glob("docs/tables/*.md")):
        text, count = MARKERS.subn(fill, path.read_text())
        if count != text.count("<!-- table:"):
            raise ValueError(f"unmatched table marker in {path.name}")
        path.write_text(text)


if __name__ == "__main__":
    main()
