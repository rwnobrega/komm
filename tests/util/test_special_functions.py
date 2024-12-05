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
    with np.testing.assert_raises(ValueError):
        komm.gaussian_q_inv(-1.0)
    with np.testing.assert_raises(ValueError):
        komm.gaussian_q_inv(2.0)
    with np.testing.assert_raises(ValueError):
        komm.gaussian_q_inv([[-1.0], [0.0], [1.0], [2.0]])


def test_gaussian_q_inverse_relation():
    x = np.linspace(-3, 3, 100)
    np.testing.assert_allclose(komm.gaussian_q_inv(komm.gaussian_q(x)), x)
    y = np.linspace(0, 1, 100)
    np.testing.assert_allclose(komm.gaussian_q(komm.gaussian_q_inv(y)), y)


def test_marcum_q():
    # fmt: off
    xs = np.arange(0, 6, step=0.1)
    expected= [
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
