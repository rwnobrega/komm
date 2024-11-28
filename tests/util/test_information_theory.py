import numpy as np
import pytest

import komm


def test_entropy_default_base():
    assert np.allclose(komm.entropy([0.5, 0.5]), 1.0)
    assert np.allclose(komm.entropy([1.0, 0.0]), 0.0)
    assert np.allclose(komm.entropy([0.25, 0.75]), 0.8112781245)
    assert np.allclose(komm.entropy([1 / 3, 1 / 3, 1 / 3]), 1.584962501)


def test_entropy_base_e():
    assert np.allclose(komm.entropy([0.5, 0.5], base="e"), 0.6931471806)
    assert np.allclose(komm.entropy([1.0, 0.0], base="e"), 0.0)
    assert np.allclose(komm.entropy([0.25, 0.75], base="e"), 0.5623351446)
    assert np.allclose(komm.entropy([1 / 3, 1 / 3, 1 / 3], base="e"), 1.098612289)


def test_entropy_base_3():
    assert np.allclose(komm.entropy([0.5, 0.5], base=3.0), 0.6309297536)
    assert np.allclose(komm.entropy([1.0, 0.0], base=3.0), 0.0)
    assert np.allclose(komm.entropy([0.25, 0.75], base=3.0), 0.5118595071)
    assert np.allclose(komm.entropy([1 / 3, 1 / 3, 1 / 3], base=3.0), 1.0)


def test_entropy_invalid_pmf():
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5, 0.5], base=3.0)
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5, 0.5], base="e")


def test_entropy_invalid_base():
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5], base=0.0)
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5], base=-1.0)
    with pytest.raises(ValueError):
        komm.entropy([0.5, 0.5], base="f")  # type: ignore


def test_relative_entropy_wikipedia():
    assert np.allclose(
        komm.relative_entropy(
            [9 / 25, 12 / 25, 4 / 25],
            [1 / 3, 1 / 3, 1 / 3],
            base="e",
        ),
        0.0852996,
    )
    assert np.allclose(
        komm.relative_entropy(
            [1 / 3, 1 / 3, 1 / 3],
            [9 / 25, 12 / 25, 4 / 25],
            base="e",
        ),
        0.097455,
    )


def test_relative_entropy_invalid_pmf():
    valid_pmf = [0.25, 0.25, 0.5]
    invalid_pmfs = [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.5, -0.5, 1.0]]
    komm.relative_entropy(valid_pmf, valid_pmf)
    for invalid_pmf in invalid_pmfs:
        with pytest.raises(ValueError):
            komm.relative_entropy(invalid_pmf, valid_pmf)
        with pytest.raises(ValueError):
            komm.relative_entropy(valid_pmf, invalid_pmf)


def test_relative_entropy_different_size():
    pmf1 = [0.25, 0.25, 0.5]
    pmf2 = [0.25, 0.75]
    komm.relative_entropy(pmf1, pmf1)
    komm.relative_entropy(pmf2, pmf2)
    with pytest.raises(ValueError):
        komm.relative_entropy(pmf1, pmf2)
    with pytest.raises(ValueError):
        komm.relative_entropy(pmf2, pmf1)


def test_relative_entropy_invalid_base():
    pmf1 = [0.25, 0.25, 0.5]
    pmf2 = [0.1, 0.1, 0.8]
    komm.relative_entropy(pmf1, pmf2, base=2.0)
    with pytest.raises(ValueError):
        komm.relative_entropy(pmf1, pmf2, base=0.0)
    with pytest.raises(ValueError):
        komm.relative_entropy(pmf1, pmf2, base=-1.0)
    with pytest.raises(ValueError):
        komm.relative_entropy(pmf1, pmf2, base="f")  # type: ignore
