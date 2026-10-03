import re
from pathlib import Path

from mkdocs.plugins import event_priority

ABBR_RE = re.compile(r"^\*\[(.+?)\]: (.*)$")


# Run before mkdocstrings copies the config.
@event_priority(100)
def on_config(config):
    glossary = {}
    for path in ["site/includes/acronyms.md", "site/includes/bibliography.md"]:
        for line in Path(path).read_text().splitlines():
            if match := ABBR_RE.match(line):
                glossary[match[1]] = match[2]
    config.mdx_configs["abbr"] = {"glossary": glossary}
