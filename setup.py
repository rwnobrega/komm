from setuptools import find_packages, setup

_long_description = """
**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by—but is not meant to be compatible with—the MATLAB® [Communications System Toolbox™](https://www.mathworks.com/help/comm/). Other sources of inspiration include [GNU Radio](https://gnuradio.org/), [CommPy](http://veeresht.info/CommPy/), and [SageMath](https://www.sagemath.org/). **Komm** is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

For installation instructions and source code, please check the [project's development page](https://github.com/rwnobrega/komm).

For library reference, please check the [project's website](https://komm.dev/).

This software is still under development. Contributions are very welcome!
"""

setup(
    name="komm",
    version="0.7.4",
    description="An open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems.",
    long_description=_long_description,
    url="https://komm.dev/",
    author="Roberto W. Nobrega",
    author_email="rwnobrega@gmail.com",
    license="GPL",
    project_urls={"Documentation": "https://komm.dev/docs/", "Source": "https://github.com/rwnobrega/komm/"},
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.8",
)
