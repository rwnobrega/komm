from itertools import product

import numpy as np
import pytest

import komm
import komm.abc

params = []

# Natural
num_bits = [1, 2, 3]
for args in product(num_bits):
    params.append(komm.NaturalLabeling(*args))

# Reflected
num_bits = [1, 2, 3]
for args in product(num_bits):
    params.append(komm.ReflectedLabeling(*args))

# Reflected rectangular
num_bits = [2, 4, (1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1)]
for args in product(num_bits):
    params.append(komm.ReflectedRectangularLabeling(*args))


@pytest.fixture(params=params, ids=lambda labeling: repr(labeling))
def labeling(request: pytest.FixtureRequest):
    return request.param


def test_labeling_equivalence_properties(labeling: komm.abc.Labeling):
    ref = komm.Labeling(labeling.matrix)
    np.testing.assert_allclose(labeling.matrix, ref.matrix)
    np.testing.assert_equal(labeling.num_bits, ref.num_bits)
    np.testing.assert_equal(labeling.cardinality, ref.cardinality)
    np.testing.assert_equal(labeling.inverse_mapping, ref.inverse_mapping)


def test_labeling_equivalence_methods(labeling: komm.abc.Labeling, rng):
    ref = komm.Labeling(labeling.matrix)
    # indices_to_bits
    indices = rng.integers(0, labeling.cardinality, size=100)
    np.testing.assert_equal(
        labeling.indices_to_bits(indices),
        ref.indices_to_bits(indices),
    )
    # bits_to_indices
    bits = rng.integers(0, 2, size=100 * labeling.num_bits)
    np.testing.assert_equal(
        labeling.bits_to_indices(bits),
        ref.bits_to_indices(bits),
    )
    # marginalize
    metrics = rng.uniform(0, 1, size=100 * labeling.cardinality)
    np.testing.assert_equal(
        labeling.marginalize(metrics),
        ref.marginalize(metrics),
    )


def test_labeling_bijective(labeling: komm.abc.Labeling, rng):
    indices = rng.integers(0, labeling.cardinality, size=100)
    np.testing.assert_equal(
        indices,
        labeling.bits_to_indices(labeling.indices_to_bits(indices)),
    )
    bits = rng.integers(0, 2, size=100 * labeling.num_bits)
    np.testing.assert_equal(
        bits,
        labeling.indices_to_bits(labeling.bits_to_indices(bits)),
    )


def test_inverse_mapping_is_cached(labeling):
    assert labeling.inverse_mapping is labeling.inverse_mapping


def test_labeling_invalid_input(labeling: komm.abc.Labeling):
    m, M = labeling.num_bits, labeling.cardinality
    for lab in [labeling, komm.Labeling(labeling.matrix)]:
        for indices in [[-1], [M]]:
            with pytest.raises(ValueError, match=rf"'indices' must be in \[0:{M}\)"):
                lab.indices_to_bits(indices)
        with pytest.raises(TypeError, match="'indices' must contain only integers"):
            lab.indices_to_bits([0.5])
        with pytest.raises(ValueError, match="'bits' must be 0 or 1"):
            lab.bits_to_indices([2] * m)
        with pytest.raises(TypeError, match="'bits' must contain only integers"):
            lab.bits_to_indices([0.5] * m)


@pytest.mark.parametrize(
    "cls, message",
    [
        (komm.NaturalLabeling, "'num_bits' must be a positive integer"),
        (komm.ReflectedLabeling, "'num_bits' must be a positive integer"),
        (komm.ReflectedRectangularLabeling, "'num_bits' must be at least 2"),
    ],
)
def test_labeling_invalid_num_bits(cls, message):
    assert type(cls(np.int64(2)).num_bits) is int
    with pytest.raises(ValueError, match=message):
        cls(0)
    with pytest.raises(TypeError, match="'num_bits' must be an integer"):
        cls(2.0)
