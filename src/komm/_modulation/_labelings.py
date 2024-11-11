import numpy as np

from komm._util.bit_operations import int2binlist
from komm._util.matrices import cartesian_product


def labeling_natural(order):
    m = order.bit_length() - 1
    labeling = np.empty((order, m), dtype=int)
    for i in range(order):
        labeling[i, :] = int2binlist(i, m)
    return labeling


def labeling_reflected(order):
    m = order.bit_length() - 1
    labeling = np.empty((order, m), dtype=int)
    for i in range(order):
        labeling[i, :] = int2binlist(i ^ (i >> 1), m)
    return labeling


def labeling_natural_2d(orders):
    order_I, order_Q = orders
    return cartesian_product(
        labeling_natural(order_I),
        labeling_natural(order_Q),
    )


def labeling_reflected_2d(orders):
    order_I, order_Q = orders
    return cartesian_product(
        labeling_reflected(order_I),
        labeling_reflected(order_Q),
    )


labelings = {
    "natural": labeling_natural,
    "reflected": labeling_reflected,
    "natural_2d": labeling_natural_2d,
    "reflected_2d": labeling_reflected_2d,
}
