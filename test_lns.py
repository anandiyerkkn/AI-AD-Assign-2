import numpy as np
import pytest

from lns import LNS8, LNS16, fp_to_lns, lns_to_fp, add, multiply, mac


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
def test_zero(fmt):
    x = fp_to_lns(0.0, fmt)
    assert x.is_zero
    assert lns_to_fp(x) == np.float32(0.0)


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
@pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
def test_basic_conversion(fmt, value):
    x = fp_to_lns(value, fmt)
    y = float(lns_to_fp(x, np.float64))
    assert np.isfinite(y)
    assert np.sign(y) == np.sign(value)


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
def test_multiply_signs(fmt):
    assert float(lns_to_fp(multiply(
        fp_to_lns(2.0, fmt), fp_to_lns(-4.0, fmt)
    ), np.float64)) < 0


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
def test_add_opposite_equal_values_gives_zero(fmt):
    a = fp_to_lns(3.0, fmt)
    b = fp_to_lns(-3.0, fmt)
    assert add(a, b).is_zero


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
def test_mac(fmt):
    a = fp_to_lns(2.0, fmt)
    b = fp_to_lns(3.0, fmt)
    c = fp_to_lns(4.0, fmt)
    result = float(lns_to_fp(mac(c, a, b), np.float64))
    assert abs(result - 10.0) < (0.5 if fmt == LNS8 else 0.005)


def test_format_order():
    assert LNS16.min_positive < LNS8.min_positive
    assert LNS16.max_magnitude > LNS8.max_magnitude


@pytest.mark.parametrize("fmt", [LNS8, LNS16])
def test_multiplication_is_log_domain(fmt):
    a = fp_to_lns(1.5, fmt)
    b = fp_to_lns(2.5, fmt)
    result = multiply(a, b)
    # Product log must be approximately sum of operand logs.
    assert result.log_magnitude == pytest.approx(
        a.log_magnitude + b.log_magnitude,
        abs=1 / fmt.scale
    )


def test_randomized_finite_results():
    rng = np.random.default_rng(123)
    values = 10.0 ** rng.uniform(-3, 3, 1000)
    for fmt in (LNS8, LNS16):
        for v in values:
            x = fp_to_lns(v, fmt)
            assert np.isfinite(float(lns_to_fp(x, np.float64)))
