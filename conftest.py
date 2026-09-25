import zlib

import numpy
import pytest

import komm


@pytest.fixture(scope="session", autouse=True)
def add_namespace(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["komm"] = komm


@pytest.fixture(autouse=True)
def rng(request: pytest.FixtureRequest):
    # Seed per test id; 42 for doctests
    if isinstance(request.node, pytest.Function):
        seed = [42, zlib.crc32(request.node.nodeid.encode())]
    else:
        seed = 42
    rng = numpy.random.default_rng(seed)
    komm.global_rng.set(rng)
    return rng
