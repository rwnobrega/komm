from random import choices, randint

import numpy as np
import pytest

import komm


def test_bits_to_int_basic():
    np.testing.assert_equal(
        komm.bits_to_int([0]),
        0,
    )
    np.testing.assert_equal(
        komm.bits_to_int([1]),
        1,
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 0, 0, 0, 1]),
        16,
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 1, 0, 1, 1]),
        26,
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 1, 0, 1, 1, 0, 0, 0]),
        26,
    )
    np.testing.assert_array_equal(
        komm.bits_to_int([[0, 0], [1, 0], [0, 1], [1, 1]]),
        [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        komm.bits_to_int([[[0, 0], [1, 0]], [[0, 1], [1, 1]]]),
        [[0, 1], [2, 3]],
    )


@pytest.mark.parametrize("n", range(1, 101))
def test_bits_to_int_big_numbers(n):
    assert komm.bits_to_int([1] * n) == 2**n - 1


def test_int_to_bits_basic():
    np.testing.assert_array_equal(
        komm.int_to_bits(0, width=1),
        [0],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits(1, width=1),
        [1],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits(16, width=5),
        [0, 0, 0, 0, 1],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits(26, width=5),
        [0, 1, 0, 1, 1],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits(26, width=8),
        [0, 1, 0, 1, 1, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits([0, 1, 2, 3], width=2),
        [[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    np.testing.assert_array_equal(
        komm.int_to_bits([[0, 1], [2, 3]], width=2),
        [[[0, 0], [1, 0]], [[0, 1], [1, 1]]],
    )


@pytest.mark.parametrize("n", range(1, 101))
def test_int_to_bits_big_numbers(n):
    np.testing.assert_array_equal(komm.int_to_bits(2**n - 1, width=n), [1] * n)


@pytest.mark.parametrize("n", [10, 100])
def test_bits_to_int_to_bits(n):
    for _ in range(1000):
        bits = choices([0, 1], k=n)
        np.testing.assert_array_equal(
            bits,
            komm.int_to_bits(komm.bits_to_int(bits), width=n),
        )


@pytest.mark.parametrize("n", [10, 100])
def test_int_to_bits_to_int(n):
    for _ in range(1000):
        integer = randint(0, 2**n - 1)
        np.testing.assert_equal(
            integer,
            komm.bits_to_int(komm.int_to_bits(integer, width=n)),
        )
