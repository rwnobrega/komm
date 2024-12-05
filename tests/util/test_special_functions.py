import numpy as np

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
