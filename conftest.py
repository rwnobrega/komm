import pytest

import numpy
import komm

@pytest.fixture(autouse=True)
def add_namespace(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['komm'] = komm
