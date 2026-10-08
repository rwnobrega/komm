import importlib
import json
import re
from pathlib import Path

# Generated text sits between these markers
MARKERS = re.compile(r"(<!-- table: (\w+) -->\n\n).*?(\n\n<!-- end table -->)", re.S)

site = Path(__file__).parent


def dumps(rows):
    # One row per line
    return "[\n" + ",\n".join("  " + json.dumps(row) for row in rows) + "\n]\n"


def fill(match):
    module = importlib.import_module(f"tables.{match[2]}")
    path = site / "docs/tables" / f"{match[2].replace('_', '-')}.json"
    # Slow tables are computed once
    if getattr(module, "cached", False) and path.exists():
        rows = json.loads(path.read_text())
    else:
        rows = module.data()
    path.write_text(dumps(rows))
    print(f"Generated {match[2]}")
    return match[1] + module.table(rows) + match[3]


def main():
    for path in sorted(site.glob("docs/tables/*.md")):
        text, count = MARKERS.subn(fill, path.read_text())
        if count != text.count("<!-- table:"):
            raise ValueError(f"unmatched table marker in {path.name}")
        path.write_text(text)


if __name__ == "__main__":
    main()
