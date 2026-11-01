# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""四层 parquet 读写：原始响应不可变，标准/因子/研究按主键去重。"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .paths import factor_dir, maps_dir, markers_dir, raw_dir, research_dir, standard_dir
from .schema import STANDARD_COLUMNS, StandardRow


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value))


def write_marker(name: str, payload: dict[str, Any], *, root: Path | None = None) -> Path:
    dest = markers_dir(root) / f"{name}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return dest


def _write_parquet(dest: Path, frame: pd.DataFrame) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    return dest


def write_raw_frame(
    source: str,
    dataset: str,
    frame: pd.DataFrame,
    *,
    request: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """原始层：保留源字段，并附加 ingestion_at 与请求参数。"""
    dest = raw_dir(root) / source / dataset / "payload.parquet"
    out = frame.copy()
    out["ingestion_at"] = datetime.now().isoformat(timespec="seconds")
    path = _write_parquet(dest, out)
    meta = dest.with_name("request.json")
    meta.write_text(
        json.dumps(
            {"source": source, "dataset": dataset, "request": request or {}, "ingestion_at": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path


def empty_standard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(STANDARD_COLUMNS))


def rows_to_frame(rows: Sequence[StandardRow]) -> pd.DataFrame:
    if not rows:
        return empty_standard_frame()
    frame = pd.DataFrame([row.as_dict() for row in rows])
    for col in ("event_date", "announcement_date", "available_at"):
        frame[col] = pd.to_datetime(frame[col]).dt.date
    return frame


def write_standard(frame: pd.DataFrame, *, name: str = "observations", root: Path | None = None) -> Path:
    dest = standard_dir(root) / f"{name}.parquet"
    out = frame.copy()
    if out.empty:
        out = empty_standard_frame()
    else:
        for col in ("event_date", "announcement_date", "available_at"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col]).dt.date
        keys = [c for c in ("dataset", "entity_id", "event_date", "field") if c in out.columns]
        out = out.drop_duplicates(subset=keys, keep="last")
        out = out.sort_values(["dataset", "entity_id", "event_date", "field"]).reset_index(drop=True)
    return _write_parquet(dest, out)


def read_standard(*, name: str = "observations", root: Path | None = None) -> pd.DataFrame:
    path = standard_dir(root) / f"{name}.parquet"
    if not path.is_file():
        return empty_standard_frame()
    return pd.read_parquet(path)


def write_factor(frame: pd.DataFrame, *, name: str, root: Path | None = None) -> Path:
    dest = factor_dir(root) / f"{name}.parquet"
    return _write_parquet(dest, frame if not frame.empty else pd.DataFrame())


def read_factor(name: str, *, root: Path | None = None) -> pd.DataFrame:
    path = factor_dir(root) / f"{name}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_research(frame: pd.DataFrame, *, name: str, root: Path | None = None) -> Path:
    dest = research_dir(root) / f"{name}.parquet"
    return _write_parquet(dest, frame if not frame.empty else pd.DataFrame())


def read_research(name: str, *, root: Path | None = None) -> pd.DataFrame:
    path = research_dir(root) / f"{name}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_map(frame: pd.DataFrame, *, name: str, root: Path | None = None) -> Path:
    dest = maps_dir(root) / f"{name}.parquet"
    return _write_parquet(dest, frame)


def read_map(name: str, *, root: Path | None = None) -> pd.DataFrame:
    path = maps_dir(root) / f"{name}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)
