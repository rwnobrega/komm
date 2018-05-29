from setuptools import setup, find_packages
from komm import __version__

_long_description = '''
**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by---but is not meant to be compatible with---the MATLAB® `Communications System Toolbox™ <https://www.mathworks.com/help/comm/>`_. Other sources of inspiration include `GNU Radio <https://gnuradio.org/>`_, `CommPy <http://veeresht.info/CommPy/>`_, and `SageMath <https://www.sagemath.org/>`_. **Komm** is licensed under the `GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

For installation instructions and source code, please check the project's `development page at GitHub <https://github.com/rwnobrega/komm>`_.

For library reference, please check the project's `documentation page at Read the Docs <http://komm.readthedocs.io/>`_.

This software is still under development.
'''

setup(
    name='komm',
    version=__version__,
    description='An open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems.',
    long_description=_long_description,
    url='https://github.com/rwnobrega/komm/',
    author='Roberto W. Nobrega',
    author_email='rwnobrega@gmail.com',
    license='GPL',
    project_urls={
        'Documentation': 'http://komm.readthedocs.io/',
        'Source': 'https://github.com/rwnobrega/komm/'},
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'scipy'],
    python_requires='>=3.5',
)
