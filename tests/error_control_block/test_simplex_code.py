import numpy as np
import pytest
from typeguard import TypeCheckError

import komm


def test_encoder():
    code = komm.SimplexCode(3)
    np.testing.assert_array_equal(
        code.encode([[1, 0, 1], [0, 0, 1]]),
        [[1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1]],
    )


def test_simplex_invalid_init():
    with pytest.raises(ValueError, match="'kappa' must be at least 2"):
        komm.SimplexCode(1)
    with pytest.raises(TypeCheckError):
        komm.SimplexCode(7 / 3)  # type: ignore
    with pytest.raises(TypeCheckError):
        komm.SimplexCode(7, 3)  # type: ignore
