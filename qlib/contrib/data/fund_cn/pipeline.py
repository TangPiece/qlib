# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""编排：stage → coverage 硬闸 → 本地 PIT → 按日 compute → 质检 → parquet/R2。"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.akshare_adapter import AkshareStatementAdapter
from .adapters.announce import StaticAnnounceAdapter
from .adapters.base import AnnounceRow, ShareCapitalAdapter, ShareCapitalRow, StatementFieldRow
from .compute import build_daily_factor_panel
from .coverage import CoverageFailed, assert_mvp_coverage, coverage_summary
from .ingest import build_pit_rows_from_staging, read_pit_rows, stage_from_adapters, write_pit_from_staging
from .paths_local import default_factor_root, default_staging_root
from .publish import publish_day
from .qa import check_day
from .staging import (
    read_announce_rows,
    read_share_capital_rows,
    read_statement_rows,
    write_marker,
    write_share_capital_rows,
)

_log = logging.getLogger(__name__)


def run_stage(
    instrument_keys: Sequence[str],
    *,
    statement_adapter=None,
    announce_adapter=None,
    statement_rows: Sequence[StatementFieldRow] | None = None,
    announce_rows: Sequence[AnnounceRow] | None = None,
    share_capital_adapter: ShareCapitalAdapter | None = None,
    share_capital_rows: Sequence[ShareCapitalRow] | None = None,
    root: Path | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """拉/注入财报、公告、股本并写入 staging。"""
    stmts = list(statement_rows) if statement_rows is not None else []
    anns = list(announce_rows) if announce_rows is not None else []
    adapter: Any = None
    if statement_rows is None:
        adapter = statement_adapter or AkshareStatementAdapter()
        stmts = adapter.fetch_statements(instrument_keys, start=start, end=end)
    if announce_rows is None:
        adapter_a = announce_adapter
        if adapter_a is None and hasattr(adapter, "fetch_announcements"):
            adapter_a = adapter
        adapter_a = adapter_a or StaticAnnounceAdapter([])
        anns = adapter_a.fetch_announcements(instrument_keys, start=start, end=end)
    info = stage_from_adapters(stmts, anns, root=root)
    shares: list[ShareCapitalRow] = list(share_capital_rows or [])
    if share_capital_rows is None and share_capital_adapter is not None:
        shares = share_capital_adapter.fetch_share_capital(instrument_keys)
    if share_capital_rows is not None or share_capital_adapter is not None:
        write_share_capital_rows(shares, root=root)
    info["share_capital_rows"] = len(shares)
    return info


def run_coverage(root: Path | None = None, *, require: bool = False) -> dict[str, Any]:
    """写 coverage marker；require=True 时 MVP 未过则抛 CoverageFailed。"""
    statements = read_statement_rows(root)
    announce = read_announce_rows(root)
    summary = coverage_summary(statements, announce)
    write_marker("coverage", summary, root=root)
    if require:
        assert_mvp_coverage(summary)
    return summary


def run_pit(root: Path | None = None) -> dict[str, Any]:
    """staging → 本地 pit/metrics.parquet。"""
    return write_pit_from_staging(root)


def _day_market(market: pd.DataFrame | None, trading_date: str) -> pd.DataFrame | None:
    if market is None or market.empty:
        return market
    iso = str(trading_date)[:10] if "-" in str(trading_date) else (
        f"{str(trading_date)[:4]}-{str(trading_date)[4:6]}-{str(trading_date)[6:8]}"
    )
    frame = market.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.date
    wanted = pd.to_datetime(iso).date()
    return frame[frame["trading_date"] == wanted].copy()


def run_compute_publish(
    trading_dates: Sequence[date | str],
    instrument_keys: Sequence[str],
    market: pd.DataFrame | None = None,
    *,
    pit_rows: Sequence[dict[str, Any]] | None = None,
    root: Path | None = None,
    factor_root: Path | None = None,
    mvp_only: bool = True,
    upload: bool = True,
    qa_dir: Path | None = None,
) -> dict[str, Any]:
    """按日算因子 → 质检 → 通过才写 parquet/R2。"""
    rows = list(pit_rows) if pit_rows is not None else read_pit_rows(root)
    shares = read_share_capital_rows(root)
    froot = factor_root or default_factor_root()
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for raw_day in trading_dates:
        iso = raw_day.isoformat() if hasattr(raw_day, "isoformat") else str(raw_day)
        day_mkt = _day_market(market, iso)
        panel = build_daily_factor_panel(
            rows,
            [iso],
            instrument_keys,
            day_mkt,
            mvp_only=mvp_only,
            share_capital=shares,
        )
        qa = check_day(
            trading_date=iso,
            wide=panel,
            pit_rows=rows,
            market=day_mkt,
            share_capital=shares,
            mvp_only=mvp_only,
            qa_dir=qa_dir,
        )
        if not qa.get("ok"):
            failed.append({"trading_date": iso, "qa": qa})
            print(f"[{iso}] qa fail: {qa.get('errors')}", flush=True)
            continue
        meta_row = publish_day(
            froot,
            iso,
            panel,
            upload_r2=upload,
            mvp_only=mvp_only,
        )
        meta_row["trading_date"] = iso
        meta_row["qa"] = qa
        published.append(meta_row)
        print(f"[{iso}] published rows={meta_row['row_count']}", flush=True)

    write_marker(
        "factor_publish",
        {
            "status": "done" if not failed else "partial",
            "days": len(published),
            "failed_days": [item["trading_date"] for item in failed],
            "row_count": int(sum(int(item.get("row_count") or 0) for item in published)),
        },
        root=root,
    )
    return {
        "days": published,
        "failed": failed,
        "row_count": int(sum(int(item.get("row_count") or 0) for item in published)),
    }


def run_pipeline(
    instrument_keys: Sequence[str],
    trading_dates: Sequence[date | str],
    *,
    statement_rows: Sequence[StatementFieldRow] | None = None,
    announce_rows: Sequence[AnnounceRow] | None = None,
    share_capital_rows: Sequence[ShareCapitalRow] | None = None,
    share_capital_adapter: ShareCapitalAdapter | None = None,
    market: pd.DataFrame | None = None,
    root: Path | None = None,
    factor_root: Path | None = None,
    mvp_only: bool = True,
    require_coverage: bool = True,
    upload: bool = True,
    qa_dir: Path | None = None,
) -> dict[str, Any]:
    """离线可注入行；直播数走 adapter。coverage 默认硬闸。"""
    staging_root = root or default_staging_root()
    stage_info = run_stage(
        instrument_keys,
        statement_rows=statement_rows,
        announce_rows=announce_rows,
        share_capital_rows=share_capital_rows,
        share_capital_adapter=share_capital_adapter,
        root=staging_root,
    )
    coverage = run_coverage(staging_root, require=require_coverage)
    pit = run_pit(staging_root)
    factors = run_compute_publish(
        trading_dates,
        instrument_keys,
        market,
        pit_rows=build_pit_rows_from_staging(staging_root),
        root=staging_root,
        factor_root=factor_root,
        mvp_only=mvp_only,
        upload=upload,
        qa_dir=qa_dir,
    )
    return {
        "stage": stage_info,
        "coverage": coverage,
        "pit": pit,
        "factors": factors,
    }


__all__ = [
    "CoverageFailed",
    "run_compute_publish",
    "run_coverage",
    "run_pipeline",
    "run_pit",
    "run_stage",
]
