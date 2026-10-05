import doctest
import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

FENCES = re.compile(r"```(pycon|python)\n(.*?)```", re.DOTALL)


def run(name):
    path = Path(__file__).parents[1] / "docs" / "recipes" / f"{name}.md"
    namespace = {}
    # Quiet, and without opening a window
    with redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"):
        for lang, code in FENCES.findall(path.read_text()):
            if lang == "pycon":
                examples = doctest.DocTestParser().get_examples(code)
                code = "".join(example.source for example in examples)
            exec(code, namespace)
    return namespace["fig"]
