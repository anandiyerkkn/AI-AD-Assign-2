"""Core LNS conversion and arithmetic.

Arithmetic is performed using signs and logarithmic magnitudes. Conversion to
ordinary floating point is only done by ``lns_to_fp`` when the caller requests
a numerical representation of an LNS result.
"""

from dataclasses import dataclass
import math
import numpy as np

from .formats import LNSFormat, LNS8, LNS16


@dataclass(frozen=True)
class LNSNumber:
    """One encoded LNS value.

    ``sign`` is -1, 0, or +1. For nonzero values, ``log_code`` stores the
    quantised log2 magnitude. Zero has sign=0 and log_code=0.
    """

    sign: int
    log_code: int
    fmt: LNSFormat

    def __post_init__(self):
        if self.sign not in (-1, 0, 1):
            raise ValueError("sign must be -1, 0, or +1")
        if not (self.fmt.min_log_code <= self.log_code <= self.fmt.max_log_code):
            raise ValueError("log_code outside selected LNS format")
        if self.sign == 0 and self.log_code != 0:
            raise ValueError("zero must have log_code=0")

    @property
    def log_magnitude(self) -> float:
        return self.log_code / self.fmt.scale

    @property
    def is_zero(self) -> bool:
        return self.sign == 0


def _as_scalar(value):
    a = np.asarray(value)
    if a.ndim != 0:
        raise ValueError("This low-level API accepts scalar values.")
    return a.item()


def fp_to_lns(value, fmt: LNSFormat = LNS16) -> LNSNumber:
    """Encode a FP32/FP16-compatible scalar into LNS.

    The source value is used only to obtain its logarithmic magnitude.
    Arithmetic is never performed in floating point after encoding.
    """
    x = _as_scalar(value)
    if not np.isfinite(x):
        raise ValueError("Only finite FP values can be represented.")

    x = float(x)
    if x == 0.0:
        return LNSNumber(0, 0, fmt)

    sign = 1 if x > 0 else -1
    log_mag = math.log2(abs(x))

    # Saturation gives explicit overflow/underflow behaviour.
    if log_mag > fmt.max_log:
        return LNSNumber(sign, fmt.max_log_code, fmt)
    if log_mag < fmt.min_log:
        return LNSNumber(sign, fmt.min_log_code, fmt)

    return LNSNumber(sign, fmt.quantize_log(log_mag), fmt)


def lns_to_fp(x: LNSNumber, dtype=np.float32):
    """Decode an LNS number to a floating-point value."""
    if x.is_zero:
        return dtype(0.0)
    value = math.ldexp(1.0, 0) * (2.0 ** x.log_magnitude)
    return dtype(x.sign * value)


def _same_format(a: LNSNumber, b: LNSNumber):
    if a.fmt != b.fmt:
        raise ValueError("Both operands must use the same LNS format.")


def _add_magnitudes(log_a: float, log_b: float) -> float:
    """log2(2^log_a + 2^log_b), without constructing either operand."""
    hi = max(log_a, log_b)
    lo = min(log_a, log_b)
    d = lo - hi
    # Correction term is evaluated in the logarithmic formula.
    return hi + math.log2(1.0 + 2.0 ** d)


def _sub_magnitudes(log_hi: float, log_lo: float) -> float:
    """log2(2^log_hi - 2^log_lo), where hi >= lo."""
    if log_hi == log_lo:
        return float("-inf")
    d = log_lo - log_hi
    return log_hi + math.log2(1.0 - 2.0 ** d)


def add(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """Add two LNS values directly in the logarithmic domain."""
    _same_format(a, b)
    fmt = a.fmt

    if a.is_zero:
        return b
    if b.is_zero:
        return a

    if a.sign == b.sign:
        log_mag = _add_magnitudes(a.log_magnitude, b.log_magnitude)
        return _encode_result(a.sign, log_mag, fmt)

    # Opposite signs: subtract smaller magnitude from larger magnitude.
    if a.log_magnitude == b.log_magnitude:
        return LNSNumber(0, 0, fmt)

    if a.log_magnitude > b.log_magnitude:
        sign, hi, lo = a.sign, a.log_magnitude, b.log_magnitude
    else:
        sign, hi, lo = b.sign, b.log_magnitude, a.log_magnitude

    log_mag = _sub_magnitudes(hi, lo)
    return _encode_result(sign, log_mag, fmt)


def multiply(a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """Multiply two LNS values directly by adding logarithms."""
    _same_format(a, b)
    fmt = a.fmt

    if a.is_zero or b.is_zero:
        return LNSNumber(0, 0, fmt)

    sign = 1 if a.sign == b.sign else -1
    log_mag = a.log_magnitude + b.log_magnitude
    return _encode_result(sign, log_mag, fmt)


def mac(acc: LNSNumber, a: LNSNumber, b: LNSNumber) -> LNSNumber:
    """Return acc + a*b, entirely in the LNS domain."""
    _same_format(acc, a)
    _same_format(acc, b)
    return add(acc, multiply(a, b))


def _encode_result(sign: int, log_mag: float, fmt: LNSFormat) -> LNSNumber:
    if log_mag == float("-inf"):
        return LNSNumber(0, 0, fmt)

    if log_mag > fmt.max_log:
        return LNSNumber(sign, fmt.max_log_code, fmt)
    if log_mag < fmt.min_log:
        # Underflow to zero is more useful for arithmetic than returning the
        # smallest representable magnitude when the true result is below it.
        return LNSNumber(0, 0, fmt)

    return LNSNumber(sign, fmt.quantize_log(log_mag), fmt)


def convert_array(values, fmt=LNS16):
    """Convert an array-like collection to a Python list of LNSNumber."""
    return [fp_to_lns(v, fmt) for v in np.asarray(values).ravel()]
