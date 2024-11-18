import pytest

from komm._lossless_coding.util import is_prefix_free, is_uniquely_decodable

test_cases = [
    {
        "code": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "uniquely_decodable": True,
        "prefix_free": True,
    },
    {
        "code": [(0,), (1,), (0, 0), (1, 1)],  # [Say06, Sec. 2.4.1, Code 2]
        "uniquely_decodable": False,
        "prefix_free": False,
    },
    {
        "code": [(0,), (1, 0), (1, 1, 0), (1, 1, 1)],  # [Say06, Sec. 2.4.1, Code 3]
        "uniquely_decodable": True,
        "prefix_free": True,
    },
    {
        "code": [(0,), (0, 1), (0, 1, 1), (0, 1, 1, 1)],  # [Say06, Sec. 2.4.1, Code 4]
        "uniquely_decodable": True,
        "prefix_free": False,
    },
    {
        "code": [(0,), (0, 1), (1, 1)],  # [Say06, Sec. 2.4.1, Code 5]
        "uniquely_decodable": True,
        "prefix_free": False,
    },
    {
        "code": [(0,), (0, 1), (1, 0)],  # [Say06, Sec. 2.4.1, Code 6]
        "uniquely_decodable": False,
        "prefix_free": False,
    },
    {
        "code": [
            (0,),
            (0, 1, 0),
            (0, 1),
            (1, 0),
        ],  # [CT06, Sec. 5.1, Table 5.1, Code 2]
        "uniquely_decodable": False,
        "prefix_free": False,
    },
    {
        "code": [
            (1, 0),
            (0, 0),
            (1, 1),
            (1, 1, 0),
        ],  # [CT06, Sec. 5.1, Table 5.1, Code 3]
        "uniquely_decodable": True,
        "prefix_free": False,
    },
    {
        "code": [
            (1,),
            (0, 1, 1),
            (0, 1, 1, 1, 0),
            (1, 1, 1, 0),
            (1, 0, 0, 1, 1),
        ],  # Wikipedia example
        "uniquely_decodable": False,
        "prefix_free": False,
    },
]


@pytest.mark.parametrize(
    "code", [case["code"] for case in test_cases if case["prefix_free"]]
)
def test_is_prefix_free(code):
    assert is_prefix_free(code)


@pytest.mark.parametrize(
    "code", [case["code"] for case in test_cases if not case["prefix_free"]]
)
def test_is_not_prefix_free(code):
    assert not is_prefix_free(code)


@pytest.mark.parametrize(
    "code", [case["code"] for case in test_cases if case["uniquely_decodable"]]
)
def test_is_uniquely_decodable(code):
    assert is_uniquely_decodable(code)


@pytest.mark.parametrize(
    "code", [case["code"] for case in test_cases if not case["uniquely_decodable"]]
)
def test_is_not_uniquely_decodable(code):
    assert not is_uniquely_decodable(code)
