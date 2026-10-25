# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把适配器行写入 staging，并从财报+公告生成本地 PIT parquet。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.base import AnnounceRow, StatementFieldRow
from .metrics import statements_to_pit_metrics
from .paths_local import pit_metrics_path
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


def build_pit_rows_from_staging(root: Path | None = None) -> list[dict[str, Any]]:
    """从 staging 现算 PIT 指标行。"""
    statements = read_statement_rows(root)
    announce = read_announce_rows(root)
    return statements_to_pit_metrics(statements, announce)


def write_pit_from_staging(root: Path | None = None) -> dict[str, Any]:
    """把 PIT 行写成本地 pit/metrics.parquet，不走 Canonical。"""
    rows = build_pit_rows_from_staging(root)
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
    version = staging_source_version(root)
    write_marker(
        "pit",
        {
            "status": "done",
            "source_version": version,
            "row_count": int(len(rows)),
            "path": str(dest),
        },
        root=root,
    )
    return {"source_version": version, "row_count": int(len(rows)), "path": str(dest)}


def read_pit_rows(root: Path | None = None) -> list[dict[str, Any]]:
    """读本地 PIT；文件不存在则从 staging 现算（不落盘）。"""
    path = pit_metrics_path(root)
    if not path.is_file():
        return build_pit_rows_from_staging(root)
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    return frame.to_dict("records")
