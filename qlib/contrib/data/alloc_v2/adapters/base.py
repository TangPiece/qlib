# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""adapter SPI：输出 StandardRow，失败只跳过本源。"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Callable, Protocol

import pandas as pd

from ..interfaces import mark_fail, mark_ok
from ..network import disable_env_proxy, is_proxy_error, prepare_network
from ..registry import DatasetDef, dataset_by_id
from ..schema import StandardRow, finite_number, first_col, in_range, parse_day, yyyymmdd

_log = logging.getLogger(__name__)


class AllocAdapter(Protocol):
    name: str

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[StandardRow]:
        ...


def lookup(dataset: str) -> DatasetDef:
    return dataset_by_id()[dataset]


def row_of(
    spec: DatasetDef,
    entity_id: str,
    day: date,
    field: str,
    value: float,
    *,
    available_at: date | None = None,
    announcement_date: date | None = None,
    evidence_class: str | None = None,
    quality_flag: str = "ok",
    source: str | None = None,
    unit: str | None = None,
) -> StandardRow:
    """构造标准行。available_at 默认等于 event_date，季报等必须显式传入。"""
    return StandardRow(
        dataset=spec.dataset,
        entity_id=entity_id,
        event_date=day,
        announcement_date=announcement_date or day,
        available_at=available_at or day,
        field=field,
        value=value,
        unit=unit or spec.unit or field,
        freq=spec.freq,
        evidence_class=evidence_class or spec.evidence_class,
        quality_flag=quality_flag,
        source=source or spec.source,
        layer=spec.layer,
    )


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
                mark_fail(label, retry_exc)
                return pd.DataFrame()
        else:
            _log.warning("adapter %s failed: %s", label, exc)
            mark_fail(label, exc)
            return pd.DataFrame()
    if frame is None:
        mark_fail(label, "empty")
        return pd.DataFrame()
    mark_ok(label, len(frame))
    return frame


def sample_weekdays(start: date | None, end: date | None, *, limit: int) -> list[date]:
    """从区间末端往前抽工作日。周末交易所份额接口会返回空表。"""
    last = end or date.today()
    first = start or (last - timedelta(days=365))
    days: list[date] = []
    cursor = last
    while cursor >= first and len(days) < limit:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def close_rows(
    frame: pd.DataFrame,
    spec: DatasetDef,
    entity_id: str,
    start: date | None,
    end: date | None,
    *,
    field: str = "close",
    evidence_class: str = "proxy",
) -> list[StandardRow]:
    if frame is None or frame.empty:
        return []
    date_col = first_col(frame, ("date", "日期", "交易时间"))
    value_col = first_col(frame, ("close", "收盘", "最新价"))
    if date_col is None or value_col is None:
        return []
    rows: list[StandardRow] = []
    for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
        day = parse_day(raw_day)
        value = finite_number(raw_value)
        if value is None or not in_range(day, start, end):
            continue
        rows.append(row_of(spec, entity_id, day, field, value, evidence_class=evidence_class))
    return rows
