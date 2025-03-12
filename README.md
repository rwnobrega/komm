# Komm

_A Python library for communication systems_.

[![PyPI page](https://badge.fury.io/py/komm.svg)](https://pypi.org/project/komm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/rwnobrega/komm/issues)

Welcome to **Komm**!

<!--intro-start-->

**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by many other communication systems libraries, such as [MATLAB® Communications System Toolbox™](https://www.mathworks.com/help/comm/), [GNU Radio](https://gnuradio.org/), [CommPy](http://veeresht.info/CommPy/), and [SageMath](https://www.sagemath.org/). **Komm** is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

<!--intro-end-->

For library reference, please check the [project's website](https://komm.dev/).

<!--notes-start-->

> [!WARNING]
> Please be advised that this project is currently under development. As such, there may be changes to the project's codebase, including the API.

<!--notes-end-->

## Installation

Before you start, make sure you have [Python](https://www.python.org/) (version 3.10 or later) installed.

### From PyPI

```bash
pip install komm
```

### From GitHub

```bash
pip install git+https://github.com/rwnobrega/komm.git@main
```

## Development

First, clone the repository:

```bash
git clone https://github.com/rwnobrega/komm
cd komm
```

This project uses [PDM](https://pdm-project.org/) for dependency management. After installing PDM, install the dependencies and activate the virtual environment:

```bash
pdm install
pdm venv activate
```

### Testing

The test suite is written using the [pytest](https://docs.pytest.org/) framework. To run the tests, execute:

```bash
pytest
```

### Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. To serve the documentation locally, run:

```bash
mkdocs serve
```

### Run demos

There are some demos available in the `demo` directory. They are written using [Streamlit](https://streamlit.io/). To run them, execute:

```bash
streamlit run demo/index.py
```

## Changelog

See the [CHANGELOG.md](CHANGELOG.md).
