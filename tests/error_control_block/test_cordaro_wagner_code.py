import numpy as np
import pytest

import komm


@pytest.mark.parametrize("n", range(2, 41))
def test_minimum_distance(n):
    r = (n + 1) // 3
    s = n - 3 * r
    if s == -1:
        (h, i, j) = (r - 1, r, r)
    elif s == 0:
        (h, i, j) = (r - 1, r, r + 1)
    else:  # s == 1:
        (h, i, j) = (r, r, r + 1)
    code = komm.CordaroWagnerCode(n)
    expected_minimum_distance = h + i
    assert code.minimum_distance == expected_minimum_distance
