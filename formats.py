"""LNS fixed-point format definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LNSFormat:
    """Definition of an LNS format.

    The sign is stored separately from the signed fixed-point logarithm.
    ``exp_bits`` is the number of bits used for the logarithmic magnitude.
    """

    name: str
    exp_bits: int
    frac_bits: int

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def max_log_code(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def min_log_code(self) -> int:
        return -(1 << (self.exp_bits - 1))

    @property
    def min_log(self) -> float:
        return self.min_log_code / self.scale

    @property
    def max_log(self) -> float:
        return self.max_log_code / self.scale

    @property
    def min_positive(self) -> float:
        return 2.0 ** self.min_log

    @property
    def max_magnitude(self) -> float:
        return 2.0 ** self.max_log

    def quantize_log(self, log_value: float) -> int:
        """Round a real logarithm to the nearest representable log code."""
        scaled = log_value * self.scale
        # Python's round uses ties-to-even, giving deterministic unbiased rounding.
        code = int(round(scaled))
        return max(self.min_log_code, min(self.max_log_code, code))


# Sign is separate, while the logarithmic field is a signed fixed-point value.
LNS8 = LNSFormat("LNS8", exp_bits=7, frac_bits=3)
LNS16 = LNSFormat("LNS16", exp_bits=15, frac_bits=9)
