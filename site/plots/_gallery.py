import doctest
import io
import re
from contextlib import chdir, redirect_stdout
from functools import cache
from pathlib import Path
from unittest.mock import patch

FENCES = re.compile(r"```(pycon|python)\n(.*?)```", re.DOTALL)

# Flag of pytest-doctestplus, unknown to doctest
doctest.register_optionflag("FLOAT_CMP")


# Run each page once for its figures
@cache
def figures(name):
    path = Path(__file__).parents[1] / "docs" / "gallery" / f"{name}.md"
    namespace = {}
    figs = []
    # Quiet, without a window, beside the page
    with (
        redirect_stdout(io.StringIO()),
        patch("matplotlib.pyplot.show"),
        chdir(path.parent),
    ):
        for lang, code in FENCES.findall(path.read_text()):
            if lang == "pycon":
                examples = doctest.DocTestParser().get_examples(code)
                code = "".join(example.source for example in examples)
            exec(code, namespace)
            if lang == "python":
                figs.append(namespace["fig"])
    return figs


def run(name, index=0):
    return figures(name)[index]
