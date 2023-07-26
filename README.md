# Komm

[![PyPI page](https://badge.fury.io/py/komm.svg)](https://pypi.org/project/komm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/rwnobrega/komm/issues)

Welcome to **Komm**!

<!--intro-start-->
**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by—but is not meant to be compatible with—the MATLAB® [Communications System Toolbox™](https://www.mathworks.com/help/comm/). Other sources of inspiration include [GNU Radio](https://gnuradio.org/), [CommPy](http://veeresht.info/CommPy/), and [SageMath](https://www.sagemath.org/). **Komm** is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
<!--intro-end-->

For library reference, please check the [project's website](https://komm.dev/).

<!--notes-start-->
Please be advised that this project is currently under development. As such, there may be changes to the project's codebase, including the API.
<!--notes-end-->

## Installation

Before you start, make sure you have [Python](https://www.python.org/) (version 3.10 or later) installed.

### From PyPI

``` bash
pip install komm
```

### From source

``` bash
git clone https://github.com/rwnobrega/komm
cd komm/
pip install .
```

## Testing

To run the tests, you need to have [pytest](https://pytest.org/) installed. Then, from the root directory of the project, run `pytest`.
