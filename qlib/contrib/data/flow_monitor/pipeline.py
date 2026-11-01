# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""编排：fetch → staging 长表 → 计算诊断面板。不写 qlib bin。"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .adapters import fetch_all
from .adapters.base import FlowRow
from .compute import run_compute
from .paths_local import default_staging_root
from .staging import coverage_from_rows, write_long_rows, write_marker

_log = logging.getLogger(__name__)


def run_fetch(
    *,
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
    rows: Sequence[FlowRow] | None = None,
) -> dict[str, Any]:
    """拉公开数据或注入测试行，写入 raw/long.parquet。"""
    root = root or default_staging_root()
    adapter_counts: dict[str, int] = {}
    if rows is None:
        fetched, adapter_counts = fetch_all(start=start, end=end)
        rows = fetched
    path = write_long_rows(rows, root=root)
    coverage = coverage_from_rows(rows, adapter_counts)
    coverage["path"] = str(path)
    write_marker("fetch", coverage, root=root)
    _log.info("flow_monitor fetch rows=%s series=%s", coverage["n_rows"], coverage["n_series"])
    return coverage


def run_pipeline(
    *,
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
    rows: Sequence[FlowRow] | None = None,
) -> dict[str, Any]:
    fetch_info = run_fetch(start=start, end=end, root=root, rows=rows)
    compute_info = run_compute(root=root)
    return {"fetch": fetch_info, "compute": compute_info}
