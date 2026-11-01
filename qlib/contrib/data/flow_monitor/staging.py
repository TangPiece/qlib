# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把 adapter 行写成统一长表 parquet，并保留 fetch 覆盖率标记。"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.base import LONG_COLUMNS, FlowRow, rows_as_dicts
from .paths_local import default_staging_root, markers_dir, raw_dir


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value))


def write_marker(name: str, payload: dict[str, Any], *, root: Path | None = None) -> Path:
    dest = markers_dir(root) / f"{name}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return dest


def write_long_rows(rows: Sequence[FlowRow], *, root: Path | None = None) -> Path:
    dest = raw_dir(root) / "long.parquet"
    frame = pd.DataFrame(rows_as_dicts(list(rows)))
    if frame.empty:
        frame = pd.DataFrame(columns=list(LONG_COLUMNS))
    else:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["asof"] = pd.to_datetime(frame["asof"]).dt.date
        frame = frame.drop_duplicates(subset=["date", "series_id"], keep="last")
        frame = frame.sort_values(["series_id", "date"]).reset_index(drop=True)
    temporary = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    return dest


def read_long_rows(root: Path | None = None) -> pd.DataFrame:
    path = raw_dir(root) / "long.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    return pd.read_parquet(path)


def coverage_from_rows(rows: Sequence[FlowRow], adapter_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """按序列统计条数与日期区间，供笔记本展示「数据不可用」。"""
    frame = pd.DataFrame(rows_as_dicts(list(rows)))
    series: dict[str, Any] = {}
    if not frame.empty:
        for series_id, part in frame.groupby("series_id"):
            days = pd.to_datetime(part["date"])
            series[str(series_id)] = {
                "rows": int(len(part)),
                "min_date": days.min().date().isoformat(),
                "max_date": days.max().date().isoformat(),
                "evidence_grade": str(part["evidence_grade"].iloc[0]),
            }
    return {
        "series": series,
        "adapter_counts": adapter_counts or {},
        "n_rows": int(len(frame)),
        "n_series": int(len(series)),
    }


def resolve_root(root: Path | None = None) -> Path:
    return root or default_staging_root()
