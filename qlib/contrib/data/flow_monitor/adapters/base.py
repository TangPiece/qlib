# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""资金流 adapter SPI：统一长表行，禁止把价格列标成 L1。"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Callable, Protocol

import pandas as pd

from ..catalog import SeriesDef, series_by_id
from .network import disable_env_proxy, is_proxy_error, prepare_network

_log = logging.getLogger(__name__)

LONG_COLUMNS = (
    "date",
    "series_id",
    "value",
    "unit",
    "asof",
    "source",
    "evidence_grade",
    "freq",
    "asset_class",
    "region",
    "investor",
    "pair_series_id",
)


@dataclass(frozen=True)
class FlowRow:
    """一条日期-序列观测。value 必须是该序列自身口径，不能把涨幅写进流量字段。"""

    date: date
    series_id: str
    value: float
    unit: str
    asof: date
    source: str
    evidence_grade: str
    freq: str
    asset_class: str
    region: str
    investor: str
    pair_series_id: str = ""


class FlowAdapter(Protocol):
    name: str

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[FlowRow]:
        ...


def rows_as_dicts(rows: list[FlowRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def parse_day(value: Any) -> date | None:
    """把各种日期表示收成 date；无法解析则返回 None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def yyyymmdd(day: date | None) -> str:
    if day is None:
        return ""
    return day.strftime("%Y%m%d")


def first_col(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """按候选名找列；先精确匹配，再包含匹配。"""
    cols = list(frame.columns)
    for name in candidates:
        if name in cols:
            return name
    for name in candidates:
        for col in cols:
            if name in str(col):
                return col
    return None


def in_range(day: date | None, start: date | None, end: date | None) -> bool:
    if day is None:
        return False
    if start is not None and day < start:
        return False
    if end is not None and day > end:
        return False
    return True


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def row_from_def(
    spec: SeriesDef,
    day: date,
    value: float,
    *,
    asof: date | None = None,
    source: str | None = None,
) -> FlowRow:
    return FlowRow(
        date=day,
        series_id=spec.series_id,
        value=value,
        unit=spec.unit,
        asof=asof or day,
        source=source or spec.source,
        evidence_grade=spec.evidence_grade,
        freq=spec.freq,
        asset_class=spec.asset_class,
        region=spec.region,
        investor=spec.investor,
        pair_series_id=spec.pair_series_id,
    )


def lookup(series_id: str) -> SeriesDef:
    return series_by_id()[series_id]


def safe_call(label: str, fn: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any) -> pd.DataFrame:
    """外网调用失败时返回空表。代理失败则改直连再试一次。"""
    prepare_network()
    try:
        frame = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 数据源失败要降级
        if is_proxy_error(exc):
            disable_env_proxy()
            _log.warning("adapter %s 代理失败，改直连重试：%s", label, exc)
            try:
                frame = fn(*args, **kwargs)
            except Exception as retry_exc:  # noqa: BLE001
                _log.warning("adapter %s failed: %s", label, retry_exc)
                return pd.DataFrame()
        else:
            _log.warning("adapter %s failed: %s", label, exc)
            return pd.DataFrame()
    if frame is None:
        return pd.DataFrame()
    return frame
