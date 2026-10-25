"""公共数值辅助。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_NAN = float("nan")


def finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else _NAN


def safe_div(num: float, den: float) -> float:
    if den is None or not math.isfinite(den) or den == 0:
        return _NAN
    if num is None or not math.isfinite(num):
        return _NAN
    return finite(num / den)


def imbalance_ratio(a: float, b: float) -> float:
    return safe_div(a - b, a + b)


def series_stat(series: pd.Series, how: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return _NAN
    if how == "mean":
        return finite(float(values.mean()))
    if how == "std":
        if len(values) < 2:
            return _NAN
        return finite(float(values.std(ddof=1)))
    if how == "max":
        return finite(float(values.max()))
    if how == "min":
        return finite(float(values.min()))
    if how == "median":
        return finite(float(values.median()))
    if how == "skew":
        if len(values) < 3:
            return _NAN
        return finite(float(values.skew()))
    if how == "kurt":
        if len(values) < 4:
            return _NAN
        return finite(float(values.kurtosis()))
    if how == "sum":
        return finite(float(values.sum()))
    if how == "count":
        return float(len(values))
    if how == "last":
        return finite(float(values.iloc[-1]))
    raise ValueError(how)


def empty_nan(keys: list[str]) -> dict[str, float]:
    return {key: _NAN for key in keys}


def clock_seconds(clock: pd.Series) -> pd.Series:
    digits = clock.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 9)
    hours = pd.to_numeric(digits.str.slice(0, 2), errors="coerce")
    minutes = pd.to_numeric(digits.str.slice(2, 4), errors="coerce")
    seconds = pd.to_numeric(digits.str.slice(4, 6), errors="coerce")
    millis = pd.to_numeric(digits.str.slice(6, 9), errors="coerce")
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return _NAN
    x = a[mask]
    y = b[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return _NAN
    return finite(float(np.corrcoef(x, y)[0, 1]))
