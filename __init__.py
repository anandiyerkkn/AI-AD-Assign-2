"""Public API for the LNS arithmetic library."""

from .formats import LNSFormat, LNS8, LNS16
from .core import LNSNumber, fp_to_lns, lns_to_fp, add, multiply, mac, convert_array
from .metrics import (
    relative_or_absolute_error,
    summarize_errors,
    conversion_errors,
    operation_errors,
    mac_errors,
    run_suite,
)

__all__ = [
    "LNSFormat", "LNS8", "LNS16",
    "LNSNumber", "fp_to_lns", "lns_to_fp",
    "add", "multiply", "mac", "convert_array",
    "relative_or_absolute_error", "summarize_errors",
    "conversion_errors", "operation_errors", "mac_errors", "run_suite",
]
