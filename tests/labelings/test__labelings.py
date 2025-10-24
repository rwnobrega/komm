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


def test_labeling_equivalence_methods(labeling: komm.abc.Labeling):
    ref = komm.Labeling(labeling.matrix)
    # indices_to_bits
    indices = np.random.randint(0, labeling.cardinality, size=100)
    np.testing.assert_equal(
        labeling.indices_to_bits(indices),
        ref.indices_to_bits(indices),
    )
    # bits_to_indices
    bits = np.random.randint(0, 2, size=100 * labeling.num_bits)
    np.testing.assert_equal(
        labeling.bits_to_indices(bits),
        ref.bits_to_indices(bits),
    )
    # marginalize
    metrics = np.random.uniform(0, 1, size=100 * labeling.cardinality)
    np.testing.assert_equal(
        labeling.marginalize(metrics),
        ref.marginalize(metrics),
    )


def test_labeling_bijective(labeling: komm.abc.Labeling):
    indices = np.random.randint(0, labeling.cardinality, size=100)
    np.testing.assert_equal(
        indices,
        labeling.bits_to_indices(labeling.indices_to_bits(indices)),
    )
    bits = np.random.randint(0, 2, size=100 * labeling.num_bits)
    np.testing.assert_equal(
        bits,
        labeling.indices_to_bits(labeling.bits_to_indices(bits)),
    )
