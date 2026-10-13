import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.VariableToFixedCode.from_sourcewords([(0,), (1, 0), (1, 1)]),
        komm.TunstallCode([0.6, 0.3, 0.1], 3),
    ],
    ids=lambda code: type(code).__name__,
)
def code(request: pytest.FixtureRequest):
    return request.param


def test_variable_to_fixed_codes_instance(code: komm.abc.VariableToFixedCode):
    assert isinstance(code, komm.abc.VariableToFixedCode)
