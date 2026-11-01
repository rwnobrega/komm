import zlib

import numpy
import pytest

import komm


@pytest.fixture(scope="session", autouse=True)
def _add_namespace(doctest_namespace):
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


@pytest.fixture(autouse=True)
def _chdir_to_page(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    # Pages read files beside them
    if request.node.path.suffix == ".md":
        monkeypatch.chdir(request.node.path.parent)
