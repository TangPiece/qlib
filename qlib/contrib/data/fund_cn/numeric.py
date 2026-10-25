"""Finite-number normalization shared by fundamental metric calculations."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_MIN_ABS_DENOMINATOR = 1e-12


def finite_float(value: Any) -> float | None:
    """Convert a scalar to finite float; missing, invalid and ±Inf become None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    min_abs_denominator: float = DEFAULT_MIN_ABS_DENOMINATOR,
) -> float | None:
    """Return a finite ratio, rejecting missing/non-finite and near-zero denominators."""
    num = finite_float(numerator)
    den = finite_float(denominator)
    if num is None or den is None or abs(den) <= min_abs_denominator:
        return None
    return finite_float(num / den)
