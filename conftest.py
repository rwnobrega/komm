import numpy
import pytest

import komm


@pytest.fixture(scope="session", autouse=True)
def add_namespace(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["komm"] = komm


@pytest.fixture(scope="function", autouse=True)
def set_global_seed():
    komm.global_rng.set(numpy.random.default_rng(seed=42))
