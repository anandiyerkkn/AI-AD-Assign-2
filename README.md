# LNS Arithmetic Library for DNN Computation

A small, importable Python library that simulates **Logarithmic Number System (LNS)** arithmetic in LNS8 and LNS16.

## Features

- FP32 / FP16 -> LNS8 / LNS16 conversion
- LNS8 / LNS16 -> FP32 conversion
- Direct logarithmic-domain:
  - addition
  - multiplication
  - multiply-accumulate (MAC)
- Positive, negative and zero operands
- Saturating overflow and underflow handling
- Round-to-nearest quantisation of the logarithmic magnitude
- Error evaluation against FP32 and FP16 references
- Random and edge-case test generation
- Importable package with `pyproject.toml`

## LNS representation used

An LNS number is represented as:

`sign × 2^L`

where `L = log2(|x|)`.

Zero is represented by a dedicated zero code.

This project uses the following fixed-point logarithm formats:

| Format | Sign | Log field | Fraction bits | Log range | Approx. magnitude range |
|---|---:|---:|---:|---:|---:|
| LNS8 | 1 bit | 7 bits | 3 | [-8, 7.875] | [2^-8, 2^7.875] |
| LNS16 | 1 bit | 15 bits | 9 | [-32, 31.998] | [2^-32, 2^31.998] |

The format is deliberately explicit because "LNS8" and "LNS16" do not uniquely determine the allocation of integer/fractional bits unless the assignment specification provides one.

## Arithmetic

### Multiplication

For nonzero values:

`log2(|xy|) = log2(|x|) + log2(|y|)`

and the signs are XORed.

### Addition

For `x,y > 0`, assuming `x >= y`:

`log2(x+y) = log2(x) + log2(1 + 2^(log2(y)-log2(x)))`

For opposite signs, subtraction is handled similarly using:

`log2(x-y) = log2(x) + log2(1 - 2^(log2(y)-log2(x)))`

The implementation operates on the stored logarithms. It does **not** convert operands to floating point and perform the requested arithmetic.

### MAC

For:

`acc <- acc + a*b`

the product is formed by adding logarithms, then the accumulator and product are combined with the LNS addition/subtraction operation.

The result is converted to floating point only when the API explicitly requests it.

## Installation

From the repository root:

```bash
pip install -e .
```

For development/testing:

```bash
pip install -e ".[test]"
pytest -q
```

## Basic use

```python
import numpy as np
from lns import LNS8, LNS16, fp_to_lns, lns_to_fp, add, multiply, mac

a = fp_to_lns(np.float32(1.5), LNS16)
b = fp_to_lns(np.float32(-2.25), LNS16)

c = multiply(a, b)
d = add(a, b)
m = mac(a, b, fp_to_lns(np.float32(4.0), LNS16))

print(lns_to_fp(c, dtype=np.float32))
print(lns_to_fp(d, dtype=np.float32))
print(lns_to_fp(m, dtype=np.float32))
```

## Error metric

For a nonzero reference:

`relative_error = |x_ref - x_lns| / |x_ref|`

For a zero reference:

`absolute_error = |x_lns|`

The evaluation module reports mean and maximum error.

## Repository structure

- `lns/formats.py` — LNS8/LNS16 format definitions and encoding helpers.
- `lns/core.py` — conversion and core LNS arithmetic.
- `lns/metrics.py` — reference generation and error calculations.
- `lns/__init__.py` — public importable API.
- `tests/test_lns.py` — correctness, edge-case and randomized tests.
- `examples/simulation.py` — assignment-style simulation and error report.
- `pyproject.toml` — package/build metadata.
- `README.md` — documentation.

## Important numerical note

LNS is not simply "floating point with a different exponent". The logarithm is quantised, so the spacing of representable magnitudes is approximately uniform in the log domain. This makes multiplication extremely cheap in hardware, but addition requires a correction operation.

LNS8 has substantially fewer bits and therefore noticeably lower precision than LNS16. The selected LNS16 format has a much wider dynamic range than LNS8, while FP32 has both a much wider range and much finer precision. FP16 has a wide dynamic range for its 16 bits because it uses a floating exponent, whereas this LNS16 simulation allocates a fixed number of fractional bits to the logarithm.

For a hardware project, the exact LNS bit allocation should be changed in `formats.py` if your course/lecturer specifies a different convention.

## GitHub submission

Create a GitHub repository, then from this directory:

```bash
git init
git add .
git commit -m "Implement LNS8 and LNS16 arithmetic library"
git branch -M main
git remote add origin https://github.com/<your-username>/lns-arithmetic.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username.
