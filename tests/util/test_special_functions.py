import numpy as np
import pytest

import komm


def test_gaussian_q_scalar():
    np.testing.assert_approx_equal(komm.gaussian_q(-np.inf), 1.0)
    np.testing.assert_approx_equal(komm.gaussian_q(-3.0), 0.998650102)
    np.testing.assert_approx_equal(komm.gaussian_q(-1.0), 0.841344746)
    np.testing.assert_approx_equal(komm.gaussian_q(0.0), 0.5)
    np.testing.assert_approx_equal(komm.gaussian_q(1.0), 0.158655254)
    np.testing.assert_approx_equal(komm.gaussian_q(3.0), 0.001349898)
    np.testing.assert_approx_equal(komm.gaussian_q(np.inf), 0.0)


def test_gaussian_q_array():
    np.testing.assert_allclose(
        komm.gaussian_q([[-1.0], [0.0], [1.0]]),
        [[0.841344746], [0.5], [0.158655254]],
    )


def test_gaussian_q_inv_scalar():
    np.testing.assert_approx_equal(komm.gaussian_q_inv(1.0), -np.inf)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.998650102), -3.0)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.841344746), -1.0)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.5), 0.0)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.158655254), 1.0)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.001349898), 3.0)
    np.testing.assert_approx_equal(komm.gaussian_q_inv(0.0), np.inf)


def test_gaussian_q_inv_array():
    np.testing.assert_allclose(
        komm.gaussian_q_inv([[0.841344746], [0.5], [0.158655254]]),
        [[-1.0], [0.0], [1.0]],
    )


def test_gaussian_q_inv_invalid_input():
    with pytest.raises(ValueError):
        komm.gaussian_q_inv(-1.0)
    with pytest.raises(ValueError):
        komm.gaussian_q_inv(2.0)
    with pytest.raises(ValueError):
        komm.gaussian_q_inv([[-1.0], [0.0], [1.0], [2.0]])


def test_gaussian_q_inverse_relation():
    x = np.linspace(-3, 3, 100)
    np.testing.assert_allclose(komm.gaussian_q_inv(komm.gaussian_q(x)), x)
    y = np.linspace(0, 1, 100)
    np.testing.assert_allclose(komm.gaussian_q(komm.gaussian_q_inv(y)), y)


@pytest.mark.parametrize(
    "x, expected",
    [
        (5.0, 2.8665157187919391167e-07),
        (8.0, 6.2209605742717841235e-16),
        (9.0, 1.1285884059538406477e-19),
        (10.0, 7.6198530241605260660e-24),
        (12.0, 1.7764821120776789977e-33),
        (20.0, 2.7536241186062336951e-89),
    ],
)
def test_gaussian_q_tail_accuracy(x, expected):
    computed = komm.gaussian_q(x)
    assert computed > 0
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_gaussian_q_tail_strictly_decreasing():
    x = np.arange(0.0, 30.0, 0.5)
    y = np.asarray(komm.gaussian_q(x))
    assert np.all(y > 0)
    assert np.all(np.diff(y) < 0)


def test_gaussian_q_inv_roundtrip_in_tail():
    for x in [1.0, 3.0, 5.0, 8.0]:
        y = komm.gaussian_q(x)
        np.testing.assert_allclose(komm.gaussian_q_inv(y), x, rtol=1e-9)


def test_marcum_q():
    # fmt: off
    xs = np.arange(0, 6, step=0.1)
    expected = [
        1.000000000000000, 0.995012479192682, 0.980198673306755, 0.955997481833100, 0.923116346386636,
        0.882496902584595, 0.835270211411272, 0.782704538241868, 0.726149037073691, 0.666976810858474,
        0.606530659712633, 0.546074426639709, 0.486752255959972, 0.429557358210739, 0.375311098851399,
        0.324652467358350, 0.278037300453194, 0.235746076555863, 0.197898699083615, 0.164474456577155,
        0.135335283236613, 0.110250525304485, 0.088921617459386, 0.071005353739637, 0.056134762834134,
        0.043936933623407, 0.034047454734599, 0.026121409853918, 0.019841094744370, 0.014920786069068,
        0.011108996538242, 0.008188701014374, 0.005976022895006, 0.004317840007633, 0.003088715408237,
        0.002187491118183, 0.001533810679324, 0.001064766236668, 0.000731802418880, 0.000497955421503,
        0.000335462627903, 0.000223745793721, 0.000147748360232, 0.000096593413722, 0.000062521503775,
        0.000040065297393, 0.000025419346516, 0.000015966783898, 0.000009929504306, 0.000006113567966,
        0.000003726653172, 0.000002249055967, 0.000001343812278, 0.000000794939362, 0.000000465571572,
        0.000000269957850, 0.000000154975314, 0.000000088081792, 0.000000049564053, 0.000000027612425,
    ]
    np.testing.assert_allclose(komm.marcum_q(1, 0.0, xs), expected)
    # fmt: on

    xs = [6, 7, 8]
    expected = [6.216524385e-03, 1.597847911e-04, 1.439441845e-06]
    np.testing.assert_allclose(komm.marcum_q(5, 2.35, xs), expected)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("a", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_marcum_q_x_zero(m, a):
    np.testing.assert_allclose(komm.marcum_q(m, a, 0.0), 1.0)


@pytest.mark.parametrize("lamb", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_marqum_q_exponential_distribution(lamb):
    # If Exp(lamb), then its complementary CDF is F(x) = Q_1(0, sqrt(2*lamb*x))
    xs = np.linspace(0, 10, 100)
    np.testing.assert_allclose(
        komm.marcum_q(1, 0.0, np.sqrt(2 * lamb * xs)), np.exp(-lamb * xs)
    )


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 2.0, 0.73532566405551922471),
        (30.0, -25.0, -24.993284651510881931),
        (40.0, 40.0, 39.306852819440054691),
        (100.0, -100.0, -99.306852819440054691),
    ],
)
def test_boxplus_values(a, b, expected):
    result = komm.boxplus(a, b)
    assert np.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_boxplus_finite():
    a = np.array([1.0, 2.0, 50.0, 80.0, -120.0, 300.0])
    b = np.array([0.0, -3.0, 50.0, -90.0, -120.0, 300.0])
    result = komm.boxplus(a, b)
    assert np.all(np.isfinite(result))


def test_boxplus_inequality():
    # Test |a ⊞ b| ≤ min(|a|, |b|)
    a = np.array([1.0, 2.0, 50.0, 80.0, -120.0, 300.0])
    b = np.array([0.0, -3.0, 50.0, -90.0, -120.0, 300.0])
    result = komm.boxplus(a, b)
    assert np.all(result <= np.minimum(np.abs(a), np.abs(b)))


def test_boxplus_identity_and_annihilator():
    # +inf is identity, 0 is annihilator
    np.testing.assert_allclose(komm.boxplus(0.0, 300.0), 0.0, atol=1e-15)
    np.testing.assert_allclose(komm.boxplus(np.inf, 3.7), 3.7, rtol=1e-12)
    np.testing.assert_allclose(komm.boxplus(-np.inf, 3.7), -3.7, rtol=1e-12)


def test_boxplus_commutative_and_sign():
    rng = np.random.default_rng()
    a = rng.normal(scale=20, size=1000)
    b = rng.normal(scale=20, size=1000)
    np.testing.assert_allclose(komm.boxplus(a, b), komm.boxplus(b, a))
    np.testing.assert_array_equal(np.sign(komm.boxplus(a, b)), np.sign(a) * np.sign(b))
