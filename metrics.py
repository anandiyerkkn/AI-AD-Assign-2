"""Reference calculations and LNS error metrics."""

import numpy as np

from .core import fp_to_lns, lns_to_fp, add, multiply, mac
from .formats import LNS8, LNS16


def relative_or_absolute_error(reference, value):
    ref = float(reference)
    val = float(value)
    if ref == 0.0:
        return abs(val)
    return abs(ref - val) / abs(ref)


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "count": int(errors.size),
    }


def conversion_errors(values, fmt, reference_dtype=np.float32):
    errors = []
    for v in values:
        ref = reference_dtype(v)
        lns = fp_to_lns(ref, fmt)
        out = lns_to_fp(lns, dtype=reference_dtype)
        errors.append(relative_or_absolute_error(ref, out))
    return summarize_errors(errors)


def operation_errors(a_values, b_values, fmt, operation, reference_dtype=np.float32):
    errors = []
    for av, bv in zip(a_values, b_values):
        a_ref = reference_dtype(av)
        b_ref = reference_dtype(bv)
        a = fp_to_lns(a_ref, fmt)
        b = fp_to_lns(b_ref, fmt)

        if operation == "add":
            lns_result = add(a, b)
            ref = reference_dtype(a_ref + b_ref)
        elif operation == "multiply":
            lns_result = multiply(a, b)
            ref = reference_dtype(a_ref * b_ref)
        else:
            raise ValueError("operation must be add or multiply")

        out = lns_to_fp(lns_result, dtype=reference_dtype)
        errors.append(relative_or_absolute_error(ref, out))
    return summarize_errors(errors)


def mac_errors(acc_values, a_values, b_values, fmt, reference_dtype=np.float32):
    errors = []
    for cv, av, bv in zip(acc_values, a_values, b_values):
        c_ref = reference_dtype(cv)
        a_ref = reference_dtype(av)
        b_ref = reference_dtype(bv)
        c = fp_to_lns(c_ref, fmt)
        a = fp_to_lns(a_ref, fmt)
        b = fp_to_lns(b_ref, fmt)
        lns_result = mac(c, a, b)
        ref = reference_dtype(c_ref + a_ref * b_ref)
        out = lns_to_fp(lns_result, dtype=reference_dtype)
        errors.append(relative_or_absolute_error(ref, out))
    return summarize_errors(errors)


def run_suite(seed=7, n_random=10000):
    """Run conversion/add/multiply/MAC simulations for both LNS formats."""
    rng = np.random.default_rng(seed)

    edge = np.array([
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
        1e-6, -1e-6, 1e-3, -1e-3, 1e3, -1e3, 1e4, -1e4
    ], dtype=np.float64)

    # Log-uniform random magnitudes expose both tiny and large values.
    mags = 10.0 ** rng.uniform(-7, 4, n_random)
    signs = rng.choice([-1.0, 1.0], n_random)
    random_values = mags * signs

    values = np.concatenate([edge, random_values])
    a = np.concatenate([edge, 10.0 ** rng.uniform(-3, 1, n_random)]) * rng.choice([-1.0, 1.0], edge.size + n_random)
    b = np.roll(a, 1)
    c = np.roll(a, 2)

    report = {}
    for fmt in (LNS8, LNS16):
        report[fmt.name] = {
            "conversion_fp32": conversion_errors(values, fmt, np.float32),
            "conversion_fp16": conversion_errors(values, fmt, np.float16),
            "add_fp32": operation_errors(a, b, fmt, "add", np.float32),
            "multiply_fp32": operation_errors(a, b, fmt, "multiply", np.float32),
            "mac_fp32": mac_errors(c, a, b, fmt, np.float32),
        }
    return report
