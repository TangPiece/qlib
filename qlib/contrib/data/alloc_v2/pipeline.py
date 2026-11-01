# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""编排：fetch → standard → factor → signal → backtest；增量 update 合并历史。"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters import fetch_all
from .backtest import run_factor_backtests
from .factors import build_all_factors
from .interfaces import dump_status
from .mappings import write_empty_maps
from .paths import default_root
from .quality import mark_quality, quality_summary
from .schema import STANDARD_COLUMNS, StandardRow
from .signals import build_signals
from .store import (
    read_standard,
    rows_to_frame,
    write_factor,
    write_marker,
    write_research,
    write_standard,
)

_log = logging.getLogger(__name__)

DEFAULT_OVERLAP_DAYS = 7
_MERGE_KEYS = ("dataset", "entity_id", "event_date", "field")


def init_maps(root: Path | None = None) -> dict[str, Any]:
    paths = write_empty_maps(root)
    write_marker("maps", {"paths": paths}, root=root)
    return {"paths": paths}


def latest_event_date(root: Path | None = None) -> date | None:
    """本地 standard 中最大 event_date；无数据返回 None。"""
    frame = read_standard(root=root)
    if frame.empty or "event_date" not in frame.columns:
        return None
    days = pd.to_datetime(frame["event_date"], errors="coerce").dropna()
    if days.empty:
        return None
    return days.max().date()


def merge_standard_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """新旧合并：同主键以 incoming 为准。"""
    if existing is None or existing.empty:
        return incoming.copy() if incoming is not None else pd.DataFrame(columns=list(STANDARD_COLUMNS))
    if incoming is None or incoming.empty:
        return existing.copy()
    combined = pd.concat([existing, incoming], ignore_index=True)
    for col in ("event_date", "announcement_date", "available_at"):
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col]).dt.date
    keys = [c for c in _MERGE_KEYS if c in combined.columns]
    combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.sort_values(["dataset", "entity_id", "event_date", "field"]).reset_index(drop=True)


def run_fetch(
    *,
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
    rows: Sequence[StandardRow] | None = None,
    merge: bool = False,
) -> dict[str, Any]:
    """拉公开数据写 standard。merge=True 时与本地历史合并，不整表覆盖。"""
    root = root or default_root()
    existing = read_standard(root=root) if merge else None
    n_before = int(len(existing)) if existing is not None and not existing.empty else 0
    if rows is None:
        fetched, counts = fetch_all(start=start, end=end)
        rows = fetched
    else:
        counts = {"injected": len(rows)}
    incoming = mark_quality(rows_to_frame(list(rows)))
    frame = merge_standard_frames(existing, incoming) if merge else incoming
    if merge:
        frame = mark_quality(frame)
    path = write_standard(frame, name="observations", root=root)
    summary = {
        "n_rows": int(len(frame)),
        "n_entities": int(frame["entity_id"].nunique()) if not frame.empty else 0,
        "n_datasets": int(frame["dataset"].nunique()) if not frame.empty else 0,
        "n_incoming": int(len(incoming)),
        "n_before": n_before,
        "merged": bool(merge),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "adapter_counts": counts,
        "quality": quality_summary(frame),
        "path": str(path),
    }
    write_marker("fetch", summary, root=root)
    dump_status(root)
    _log.info(
        "alloc_v2 fetch rows=%s incoming=%s merged=%s datasets=%s",
        summary["n_rows"],
        summary["n_incoming"],
        merge,
        summary["n_datasets"],
    )
    return summary


class UpdateError(RuntimeError):
    """增量更新前置条件不满足（例如本地无 standard）。"""


def run_update(
    *,
    root: Path | None = None,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    end: date | None = None,
    rows: Sequence[StandardRow] | None = None,
) -> dict[str, Any]:
    """从本地最近 event_date 重叠窗口增量拉取并合并，再重算因子/信号/回测。"""
    root = root or default_root()
    last = latest_event_date(root)
    if last is None:
        raise UpdateError("本地无 standard 数据，请先运行：./scripts/run_alloc_v2.sh all --start 2018-01-01")
    end_day = end or date.today()
    start_day = last - timedelta(days=max(0, int(overlap_days)))
    fetch_info = run_fetch(start=start_day, end=end_day, root=root, rows=rows, merge=True)
    factor_info = run_factor(root=root)
    signal_info = run_signal(root=root)
    backtest_info = run_backtest(root=root)
    summary = {
        "last_event_date": last.isoformat(),
        "overlap_days": int(overlap_days),
        "start": start_day.isoformat(),
        "end": end_day.isoformat(),
        "fetch": fetch_info,
        "factor": factor_info,
        "signal": signal_info,
        "backtest": backtest_info,
    }
    write_marker("update", summary, root=root)
    return summary


def run_standard(root: Path | None = None) -> dict[str, Any]:
    """标准层已在 fetch 写入；此处重跑质量标记。"""
    frame = mark_quality(read_standard(root=root))
    path = write_standard(frame, name="observations", root=root)
    summary = {"path": str(path), "quality": quality_summary(frame), "n_rows": int(len(frame))}
    write_marker("standard", summary, root=root)
    return summary


def run_factor(root: Path | None = None) -> dict[str, Any]:
    standard = read_standard(root=root)
    factors = build_all_factors(standard)
    path = write_factor(factors, name="all", root=root)
    summary = {
        "n_rows": int(len(factors)),
        "n_factors": int(factors["factor_id"].nunique()) if not factors.empty else 0,
        "path": str(path),
    }
    write_marker("factor", summary, root=root)
    return summary


def run_signal(root: Path | None = None) -> dict[str, Any]:
    from .store import read_factor

    standard = read_standard(root=root)
    factors = read_factor("all", root=root)
    payload = build_signals(factors, standard)
    paths = {
        "S1": write_research(pd.DataFrame([payload["S1"]]), name="signal_s1", root=root),
        "S2": write_research(payload["S2"], name="signal_s2", root=root),
        "S3": write_research(payload["S3"], name="signal_s3", root=root),
        "S4": write_research(payload["S4"], name="signal_s4", root=root),
        "invalidation": write_research(payload["invalidation"], name="signal_invalidation", root=root),
    }
    summary = {
        "S1": payload["S1"].get("value"),
        "n_s2": int(len(payload["S2"])) if isinstance(payload["S2"], pd.DataFrame) else 0,
        "n_s4": int(len(payload["S4"])) if isinstance(payload["S4"], pd.DataFrame) else 0,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    write_marker("signal", summary, root=root)
    return summary


def run_backtest(root: Path | None = None) -> dict[str, Any]:
    from .store import read_factor

    standard = read_standard(root=root)
    factors = read_factor("all", root=root)
    results = run_factor_backtests(factors, standard)
    path = write_research(results, name="backtest_factors", root=root)
    summary = {
        "n_results": int(len(results)),
        "n_failed": int((results["status"] == "failed").sum()) if not results.empty and "status" in results.columns else 0,
        "path": str(path),
    }
    write_marker("backtest", summary, root=root)
    return summary


def run_pipeline(
    *,
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
    rows: Sequence[StandardRow] | None = None,
) -> dict[str, Any]:
    root = root or default_root()
    maps = init_maps(root)
    fetch_info = run_fetch(start=start, end=end, root=root, rows=rows)
    standard_info = run_standard(root=root)
    factor_info = run_factor(root=root)
    signal_info = run_signal(root=root)
    backtest_info = run_backtest(root=root)
    return {
        "maps": maps,
        "fetch": fetch_info,
        "standard": standard_info,
        "factor": factor_info,
        "signal": signal_info,
        "backtest": backtest_info,
    }
