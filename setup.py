from setuptools import setup, find_packages
from komm import __version__


setup(
    name='komm',
    version=__version__,
    description='An open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems.',
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
