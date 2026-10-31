import ast
import doctest
import re
from pathlib import Path

from properdocs.plugins import event_priority
from properdocs.structure.files import File

ABBR_RE = re.compile(r"^\*\[(.+?)\]: (.*)$")
DOCTEST_FLAGS_RE = re.compile(r"[ \t]*#[ \t]*doctest:.+$", re.MULTILINE)
FENCE_RE = re.compile(r"```(pycon|python)\n(.*?)```", re.DOTALL)
TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)

# As the source button in the reference
SCRIPT_BUTTON = (
    '<div><a href="{}" title="Download script" class="md-content__button md-icon"'
    ' download><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path'
    ' d="M5 20h14v-2H5m14-9h-4V3H9v6H5l7 7z"/></svg></a></div>\n'
)

# Flag of pytest-doctestplus, unknown to doctest
doctest.register_optionflag("FLOAT_CMP")


# Run before mkdocstrings copies the config.
@event_priority(100)
def on_config(config):
    glossary = {}
    for path in ["site/includes/acronyms.md", "site/includes/bibliography.md"]:
        for line in Path(path).read_text().splitlines():
            if match := ABBR_RE.match(line):
                glossary[match[1]] = match[2]
    config.mdx_configs["abbr"] = {"glossary": glossary}


# Hide doctest flags, as in docstrings.
def on_page_markdown(markdown, **kwargs):
    return DOCTEST_FLAGS_RE.sub("", markdown)


# Print values, as the REPL does.
def print_value(source):
    match ast.parse(source).body:
        case [ast.Expr() as stmt] if not source.startswith("print("):
            expr = ast.get_source_segment(source, stmt)
            return f"print({expr}){source[len(expr):]}"
    return source


def strip_prompts(code):
    parts = []
    end = 0
    for example in doctest.DocTestParser().get_examples(code):
        # Keep blank lines between examples
        if parts and example.lineno > end:
            parts.append("\n")
        source = DOCTEST_FLAGS_RE.sub("", example.source)
        parts.append(print_value(source) if example.want else source)
        end = example.lineno + example.source.count("\n") + example.want.count("\n")
    return "".join(parts)


# One cell per code block.
def script(markdown, url):
    cells = [f"# {TITLE_RE.search(markdown)[1]}\n# {url}\n"]
    for lang, code in FENCE_RE.findall(markdown):
        cells.append("# %%\n" + (strip_prompts(code) if lang == "pycon" else code))
    return "\n".join(cells)


# Offer gallery pages as scripts.
def on_files(files, config):
    for file in files.documentation_pages():
        if file.src_uri.startswith("gallery/") and file.name != "index":
            uri = file.src_uri.removesuffix(".md") + ".py"
            content = script(file.content_string, config.site_url + file.url)
            files.append(File.generated(config, uri, content=content))


# Link gallery pages to their scripts.
def on_page_content(html, page, files, **kwargs):
    uri = page.file.src_uri.removesuffix(".md") + ".py"
    if file := files.get_file_from_path(uri):
        html = SCRIPT_BUTTON.format(file.url_relative_to(page.file)) + html
    return html
