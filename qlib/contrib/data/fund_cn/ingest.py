# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把适配器行写入 staging，并从财报+公告生成本地 PIT parquet（含按年分片）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.base import AnnounceRow, StatementFieldRow
from .coverage import drop_statements_without_announce, filter_statements_by_report_min
from .metrics import statements_to_pit_metrics
from .paths_local import pit_metrics_path, pit_year_dir, pit_year_path
from .source_policy import COVERAGE_REPORT_MIN_DATE, RESTATEMENT_UNRECOVERABLE
from .staging import (
    read_announce_rows,
    read_statement_rows,
    staging_source_version,
    write_announce_rows,
    write_marker,
    write_statement_rows,
)


def stage_from_adapters(
    statement_rows: Sequence[StatementFieldRow],
    announce_rows: Sequence[AnnounceRow],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """写入 statements/announce parquet，并记 stage marker。"""
    stmt_path = write_statement_rows(statement_rows, root=root)
    ann_path = write_announce_rows(announce_rows, root=root)
    version = staging_source_version(root)
    write_marker(
        "stage",
        {
            "status": "done",
            "source_version": version,
            "statement_rows": len(statement_rows),
            "announce_rows": len(announce_rows),
            "statements_path": str(stmt_path),
            "announce_path": str(ann_path),
        },
        root=root,
    )
    return {
        "source_version": version,
        "statement_rows": len(statement_rows),
        "announce_rows": len(announce_rows),
        "statements_path": str(stmt_path),
        "announce_path": str(ann_path),
    }


def prepare_statements_for_pit(
    statements: pd.DataFrame,
    announce: pd.DataFrame,
    *,
    report_min: date | None = COVERAGE_REPORT_MIN_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """窗口化 + 剔除无公告修订行，供 PIT / coverage 共用。

    返回 (保留科目, 缺失清单, 统计)。
    """
    windowed = filter_statements_by_report_min(statements, report_min=report_min)
    kept, missing = drop_statements_without_announce(windowed, announce)
    stats = {
        "report_min": str(report_min) if report_min else None,
        "input_rows": int(len(statements)) if statements is not None else 0,
        "windowed_rows": int(len(windowed)) if windowed is not None else 0,
        "kept_rows": int(len(kept)),
        "dropped_missing_announce_keys": int(len(missing)),
        "quality_tag": RESTATEMENT_UNRECOVERABLE,
    }
    return kept, missing, stats


def build_pit_rows_from_staging(
    root: Path | None = None,
    *,
    report_min: date | None = COVERAGE_REPORT_MIN_DATE,
    drop_unannounced: bool = True,
) -> list[dict[str, Any]]:
    """从 staging 现算 PIT 指标行（可选窗口化并丢弃无公告修订）。"""
    statements = read_statement_rows(root)
    announce = read_announce_rows(root)
    if drop_unannounced:
        statements, _, _ = prepare_statements_for_pit(
            statements, announce, report_min=report_min
        )
    elif report_min is not None:
        statements = filter_statements_by_report_min(statements, report_min=report_min)
    return statements_to_pit_metrics(statements, announce)


def _write_pit_year_shards(frame: pd.DataFrame, root: Path | None) -> list[str]:
    """按 available_time 年份落 by_year/{YYYY}.parquet；返回已写年份列表。"""
    if frame is None or frame.empty or "available_time" not in frame.columns:
        return []
    out = frame.copy()
    out["_ay"] = pd.to_datetime(out["available_time"], utc=True).dt.year
    written: list[str] = []
    year_root = pit_year_dir(root)
    # 重建分片前清空旧年文件，避免残留年份污染
    for old in year_root.glob("*.parquet"):
        old.unlink(missing_ok=True)
    for year, group in out.groupby("_ay", sort=True):
        dest = pit_year_path(root, int(year))
        temporary = dest.with_suffix(".parquet.tmp")
        group.drop(columns=["_ay"]).to_parquet(temporary, index=False)
        temporary.replace(dest)
        written.append(str(int(year)))
    return written


def write_pit_from_staging(
    root: Path | None = None,
    *,
    report_min: date | None = COVERAGE_REPORT_MIN_DATE,
) -> dict[str, Any]:
    """把 PIT 行写成本地 pit/metrics.parquet + by_year 分片。"""
    statements = read_statement_rows(root)
    announce = read_announce_rows(root)
    kept, missing, prep = prepare_statements_for_pit(
        statements, announce, report_min=report_min
    )
    # 落缺失清单，便于 EM 重试与审计
    write_marker(
        "missing_announce",
        {
            "count": int(len(missing)),
            "report_min": prep.get("report_min"),
            # 清单截断，避免 marker 过大；全量 count 仍保留
            "keys": missing.head(5000).to_dict("records") if not missing.empty else [],
        },
        root=root,
    )
    rows = statements_to_pit_metrics(kept, announce)
    dest = pit_metrics_path(root)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "instrument_key",
                "metric_code",
                "available_time",
                "report_period_end",
                "revision",
                "value",
            ]
        )
    temporary = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    years = _write_pit_year_shards(frame, root)
    version = staging_source_version(root)
    payload = {
        "status": "done",
        "source_version": version,
        "row_count": int(len(rows)),
        "path": str(dest),
        "year_shards": years,
        "prepare": prep,
        "quality_tags": [RESTATEMENT_UNRECOVERABLE],
    }
    write_marker("pit", payload, root=root)
    return payload


def read_pit_frame(
    root: Path | None = None,
    *,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """读 PIT：优先按年分片拼接；无分片则回退 metrics.parquet。

    ``years`` 为交易年时建议传入 ``[Y-2, Y-1, Y]``，避免全量载入。
    """
    empty_cols = [
        "instrument_key",
        "metric_code",
        "available_time",
        "report_period_end",
        "revision",
        "value",
    ]
    if years:
        frames: list[pd.DataFrame] = []
        for year in sorted({int(y) for y in years}):
            path = pit_year_path(root, year)
            if path.is_file():
                frames.append(pd.read_parquet(path))
        if frames:
            return pd.concat(frames, ignore_index=True)
    path = pit_metrics_path(root)
    if path.is_file():
        return pd.read_parquet(path)
    rows = build_pit_rows_from_staging(root)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_cols)


def read_pit_rows(
    root: Path | None = None,
    *,
    years: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """读本地 PIT；可按年分片过滤。"""
    frame = read_pit_frame(root, years=years)
    if frame.empty:
        return []
    return frame.to_dict("records")


def pit_years_for_trade_year(trade_year: int, *, lookback_years: int = 2) -> list[int]:
    """交易年 Y 所需 PIT 分片年份：Y-lookback … Y（含公告滞后）。"""
    start = int(trade_year) - int(lookback_years)
    return list(range(start, int(trade_year) + 1))
