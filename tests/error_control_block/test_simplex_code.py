import numpy as np
import pytest

import komm


def test_encoder():
    code = komm.SimplexCode(3)
    np.testing.assert_equal(
        code.encode([[1, 0, 1], [0, 0, 1]]),
        [[1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1]],
    )


def test_simplex_invalid_init():
    with pytest.raises(ValueError, match="'mu' must be at least 2"):
        komm.SimplexCode(1)
    with pytest.raises(TypeError, match="'mu' must be an integer"):
        komm.SimplexCode(7 / 3)  # type: ignore
    with pytest.raises(TypeError, match="takes 2 positional arguments"):
        komm.SimplexCode(7, 3)  # type: ignore
    with pytest.raises(TypeError, match="'lengthened' must be a boolean"):
        komm.SimplexCode(3, lengthened="False")  # type: ignore


@pytest.mark.parametrize("information_set", ["middle", [0, 1, 2]])
def test_simplex_invalid_information_set(information_set):
    with pytest.raises(ValueError, match="'information_set' must be 'left' or 'right'"):
        komm.SimplexCode(3, information_set=information_set)


@pytest.mark.parametrize("mu", range(2, 7))
@pytest.mark.parametrize("lengthened", [False, True])
@pytest.mark.parametrize("side, opposite", [("left", "right"), ("right", "left")])
def test_simplex_hamming_duality(mu, lengthened, side, opposite):
    simplex = komm.SimplexCode(mu, lengthened=lengthened, information_set=side)
    hamming = komm.HammingCode(mu, extended=lengthened, information_set=opposite)
    np.testing.assert_equal(simplex.generator_matrix, hamming.check_matrix)
    np.testing.assert_equal(simplex.check_matrix, hamming.generator_matrix)
