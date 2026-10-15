import doctest
import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

FENCES = re.compile(r"```(pycon|python)\n(.*?)```", re.DOTALL)


def run(name, index=0):
    path = Path(__file__).parents[1] / "docs" / "recipes" / f"{name}.md"
    namespace = {}
    figs = []
    # Quiet, and without opening a window
    with redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"):
        for lang, code in FENCES.findall(path.read_text()):
            if lang == "pycon":
                examples = doctest.DocTestParser().get_examples(code)
                code = "".join(example.source for example in examples)
            exec(code, namespace)
            if lang == "python":
                figs.append(namespace["fig"])
    fig = figs.pop(index)
    for other in figs:
        plt.close(other)
    return fig
