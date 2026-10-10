from re import escape

import pytest

import komm


@pytest.mark.parametrize("width", [0.0, -1.0, 1.5])
def test_rectangular_pulse_invalid_width(width):
    message = f"'width' must satisfy 0 < width <= 1 (got {width})"
    with pytest.raises(ValueError, match=escape(message)):
        komm.RectangularPulse(width=width)


def test_rectangular_pulse_width_not_real():
    with pytest.raises(TypeError, match=r"'width' must be a real number \(got str\)"):
        komm.RectangularPulse(width="1")  # type: ignore
