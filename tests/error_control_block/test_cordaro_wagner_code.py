import pytest
from typeguard import TypeCheckError

import komm


@pytest.mark.parametrize("n", range(2, 41))
def test_cordaro_wagner_minimum_distance(n):
    r = (n + 1) // 3
    s = n - 3 * r
    if s == -1:
        (h, i, j) = (r - 1, r, r)
    elif s == 0:
        (h, i, j) = (r - 1, r, r + 1)
    else:  # s == 1:
        (h, i, j) = (r, r, r + 1)
    code = komm.CordaroWagnerCode(n)
    assert code.minimum_distance() == h + i


def test_cordaro_wagner_code_invalid_init():
    with pytest.raises(ValueError, match="'n' must be at least 2"):
        komm.CordaroWagnerCode(0)
    with pytest.raises(ValueError, match="'n' must be at least 2"):
        komm.CordaroWagnerCode(-1)
    with pytest.raises(TypeCheckError):
        komm.CordaroWagnerCode(3.0)  # type: ignore
    with pytest.raises(TypeCheckError):
        komm.CordaroWagnerCode("3")  # type: ignore
