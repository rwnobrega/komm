import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "length, generator_polynomial, parity_check_polynomial",
    [
        (7, 0b1011, 0b10111),  # Hamming (7, 4)
        (23, 0b110001110101, 0b1111100100101),  # Golay (23, 12)
    ],
)
def test_cyclic_code(length, generator_polynomial, parity_check_polynomial):
    code_g = komm.CyclicCode.from_generator_polynomial(length, generator_polynomial)
    code_h = komm.CyclicCode.from_parity_check_polynomial(length, parity_check_polynomial)
    assert code_g.parity_check_polynomial == parity_check_polynomial
    assert code_h.generator_polynomial == generator_polynomial
