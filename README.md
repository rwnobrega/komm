# Komm

[![PyPI page](https://badge.fury.io/py/komm.svg)](https://pypi.org/project/komm/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rwnobrega/komm/master?filepath=demo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black/)

Welcome to **Komm**'s development page!

**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by---but is not meant to be compatible with---the MATLAB® [Communications System Toolbox™](https://www.mathworks.com/help/comm/). Other sources of inspiration include [GNU Radio](https://gnuradio.org/), [CommPy](http://veeresht.info/CommPy/), and [SageMath](https://www.sagemath.org/). **Komm** is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

For library reference, please check the project's [documentation page](https://komm.dev/docs/).

This software is still under development. Contributions are very welcome!

## Installation

Before you start, make sure you have [Python](https://www.python.org/) (version 3.8 or later), [NumPy](https://www.numpy.org/), and [SciPy](https://www.scipy.org/) installed.

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

To run the tests, you need to have [pytest](https://pytest.org/) installed. Then, from the root directory of the project, run:

``` bash
python -m pytest komm/ --doctest-modules
python -m pytest tests/
```
