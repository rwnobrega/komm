import numpy as np
import pytest

import komm


def test_general_modulation():
    mod = komm.Modulation(
        constellation=[1, 2, 3, 4],
        labeling=[[0, 0], [0, 1], [1, 1], [1, 0]],
    )
    np.testing.assert_array_equal(mod.constellation, [1, 2, 3, 4])
    np.testing.assert_array_equal(mod.labeling, [[0, 0], [0, 1], [1, 1], [1, 0]])
    np.testing.assert_array_equal(mod.bits_per_symbol, 2)
    np.testing.assert_array_equal(mod.energy_per_symbol, 7.5)
    np.testing.assert_array_equal(mod.energy_per_bit, 3.75)
    np.testing.assert_array_equal(mod.symbol_mean, 2.5)
    np.testing.assert_array_equal(mod.minimum_distance, 1.0)


def test_general_modulation_invalid():
    # Valid constellation and labeling
    komm.Modulation(
        constellation=[1, 2, 3, 4],
        labeling=[[0, 0], [0, 1], [1, 1], [1, 0]],
    )
    # Constellation order is not a power of 2
    with pytest.raises(ValueError):
        komm.Modulation(
            constellation=[1, 2, 3],
            labeling=[[0, 0], [0, 1], [1, 0]],
        )
    # Invalid shape of labeling
    with pytest.raises(ValueError):
        komm.Modulation(
            constellation=[1, 2, 3, 4],
            labeling=[[0, 0], [0, 1], [1, 0]],
        )
    # Non binary labeling
    with pytest.raises(ValueError):
        komm.Modulation(
            constellation=[1, 2, 3, 4],
            labeling=[[0, 0], [0, 1], [1, 2], [1, 0]],
        )
    # Non distinct labeling
    with pytest.raises(ValueError):
        komm.Modulation(
            constellation=[1, 2, 3, 4],
            labeling=[[0, 0], [0, 0], [1, 1], [1, 0]],
        )


def test_general_modulation_modulate():
    mod = komm.Modulation(
        constellation=[1, 2, 3, 4],
        labeling=[[0, 0], [0, 1], [1, 1], [1, 0]],
    )
    np.testing.assert_array_equal(mod.modulate([0, 0]), 1)
    np.testing.assert_array_equal(mod.modulate([0, 1]), 2)
    np.testing.assert_array_equal(mod.modulate([1, 1]), 3)
    np.testing.assert_array_equal(mod.modulate([1, 0]), 4)
    np.testing.assert_array_equal(mod.modulate([0, 0, 0, 1]), [1, 2])


def test_general_modulation_demodulate_hard():
    mod = komm.Modulation(
        constellation=[1, 2, 3, 4],
        labeling=[[0, 0], [0, 1], [1, 1], [1, 0]],
    )
    np.testing.assert_array_equal(mod.demodulate_hard([1, 2]), [0, 0, 0, 1])
