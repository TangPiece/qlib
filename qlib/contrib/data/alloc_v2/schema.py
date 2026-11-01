# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2 统一表结构：原始、标准、因子、研究四层字段。

周/月/季禁止伪装成日频。估算必须打 evidence_class=estimate。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from typing import Any, Literal

SERIES_SET = "alloc_research@v2"
SERIES_SET_CODE = "alloc_research"
SERIES_SET_VERSION = "v2"
SCHEMA_VERSION = "v2"

EvidenceClass = Literal["direct", "proxy", "estimate"]
Frequency = Literal["day", "week", "month", "quarter", "event"]
SourceGrade = Literal["A", "B", "C", "D"]
QualityFlag = Literal["ok", "missing", "stale", "inconsistent", "unavailable", "reference_only"]
StatementKind = Literal["fact", "estimate", "proxy", "inference", "signal"]

STANDARD_COLUMNS = (
    "dataset",
    "entity_id",
    "event_date",
    "announcement_date",
    "available_at",
    "field",
    "value",
    "unit",
    "freq",
    "evidence_class",
    "quality_flag",
    "version",
    "source",
    "layer",
)

FACTOR_COLUMNS = STANDARD_COLUMNS + (
    "factor_id",
    "window",
    "formula_version",
)

RESEARCH_COLUMNS = (
    "asof",
    "entity_id",
    "signal_id",
    "value",
    "statement_kind",
    "available_at",
    "quality_flag",
    "notes",
    "version",
)


@dataclass(frozen=True)
class StandardRow:
    """标准层一行：经济日、披露日、可观测日必须分开。"""

    dataset: str
    entity_id: str
    event_date: date
    announcement_date: date
    available_at: date
    field: str
    value: float
    unit: str
    freq: str
    evidence_class: str
    quality_flag: str = "ok"
    version: str = SCHEMA_VERSION
    source: str = ""
    layer: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def standard_field_names() -> tuple[str, ...]:
    return tuple(item.name for item in fields(StandardRow))


def parse_day(value: Any) -> date | None:
    """把各种日期表示收成 date；无法解析则返回 None。"""
    if value is None:
        return None
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if pd is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def yyyymmdd(day: date | None) -> str:
    if day is None:
        return ""
    return day.strftime("%Y%m%d")


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
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd is not None and pd.isna(number):
        return None
    return number


def first_col(frame: Any, candidates: tuple[str, ...]) -> str | None:
    """按候选名找列；先精确匹配，再包含匹配。"""
    cols = list(getattr(frame, "columns", []))
    for name in candidates:
        if name in cols:
            return name
    for name in candidates:
        for col in cols:
            if name in str(col):
                return col
    return None
