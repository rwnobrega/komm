from random import choices, randint

import numpy as np
import pytest

import komm


def test_bits_to_int_basic():
    np.testing.assert_equal(
        komm.bits_to_int([0], width=1),
        [0],
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 0, 0, 0, 1], width=5),
        [16],
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 1, 0, 1, 1, 0, 0, 0], width=8),
        [26],
    )
    np.testing.assert_equal(
        komm.bits_to_int([0, 1, 0, 1, 1, 0, 0, 0], width=4),
        [10, 1],
    )
    np.testing.assert_equal(
        komm.bits_to_int([[0, 0, 1, 0], [0, 1, 1, 1]], width=2),
        [[0, 1], [2, 3]],
    )


def test_int_to_bits_basic():
    np.testing.assert_equal(
        komm.int_to_bits([0], width=1),
        [0],
    )
    np.testing.assert_equal(
        komm.int_to_bits([16], width=5),
        [0, 0, 0, 0, 1],
    )
    np.testing.assert_equal(
        komm.int_to_bits([26], width=8),
        [0, 1, 0, 1, 1, 0, 0, 0],
    )
    np.testing.assert_equal(
        komm.int_to_bits([10, 1], width=4),
        [0, 1, 0, 1, 1, 0, 0, 0],
    )
    np.testing.assert_equal(
        komm.int_to_bits([[0, 1], [2, 3]], width=2),
        [[0, 0, 1, 0], [0, 1, 1, 1]],
    )


def test_bit_operations_bit_order():
    np.testing.assert_equal(
        komm.bits_to_int([0, 0, 0, 0, 1, 0], width=6, bit_order="MSB-first"),
        [2],
    )
    np.testing.assert_equal(
        komm.int_to_bits([2], width=6, bit_order="MSB-first"),
        [0, 0, 0, 0, 1, 0],
    )


def test_bit_operations_invalid():
    with pytest.raises(ValueError):
        komm.bits_to_int([0, 1, 0], width=2)
    with pytest.raises(ValueError):
        komm.bits_to_int([0, 2], width=2)
    with pytest.raises(ValueError):
        komm.bits_to_int([0, 1], width=0)
    with pytest.raises(ValueError):
        komm.bits_to_int([0, 1], width=64)
    with pytest.raises(ValueError):
        komm.int_to_bits([4], width=2)
    with pytest.raises(ValueError):
        komm.int_to_bits([-1], width=2)
    with pytest.raises(ValueError):
        komm.int_to_bits([0], width=1, bit_order="invalid")  # type: ignore


@pytest.mark.parametrize("width", range(1, 64))
def test_bits_to_int_to_bits(width):
    for _ in range(100):
        bits = choices([0, 1], k=3 * width)
        np.testing.assert_equal(
            bits,
            komm.int_to_bits(komm.bits_to_int(bits, width), width),
        )


@pytest.mark.parametrize("width", range(1, 64))
def test_int_to_bits_to_int(width):
    for _ in range(100):
        integers = [randint(0, 2**width - 1) for _ in range(3)]
        np.testing.assert_equal(
            integers,
            komm.bits_to_int(komm.int_to_bits(integers, width), width),
        )


def test_binary_basic():
    assert komm.to_binary(0) == []
    assert komm.to_binary(0, width=1) == [0]
    assert komm.to_binary(26) == [0, 1, 0, 1, 1]
    assert komm.to_binary(26, width=8) == [0, 1, 0, 1, 1, 0, 0, 0]
    assert komm.to_binary(26, bit_order="MSB-first") == [1, 1, 0, 1, 0]
    assert komm.from_binary([]) == 0
    assert komm.from_binary([0, 1, 0, 1, 1]) == 26
    assert komm.from_binary([1, 1, 0, 1, 0], bit_order="MSB-first") == 26


def test_binary_invalid():
    with pytest.raises(ValueError):
        komm.from_binary([0, 2])
    with pytest.raises(TypeError):
        komm.to_binary(1.5)  # type: ignore


@pytest.mark.parametrize("width", range(1, 201))
def test_binary_big_numbers(width):
    assert komm.from_binary([1] * width) == 2**width - 1
    assert komm.to_binary(2**width - 1, width=width) == [1] * width


@pytest.mark.parametrize("width", [10, 100, 200])
def test_binary_round_trip(width):
    for _ in range(100):
        integer = randint(0, 2**width - 1)
        assert komm.from_binary(komm.to_binary(integer, width=width)) == integer
