# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""编排：stage → coverage 硬闸 → 本地 PIT → 按日 compute → 质检 → parquet/R2。

默认源策略：国泰安科目为主，EM NOTICE 补公告，免费源仅抽样验证。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.akshare_adapter import AkshareStatementAdapter
from .adapters.announce import StaticAnnounceAdapter
from .adapters.base import AnnounceRow, ShareCapitalAdapter, ShareCapitalRow, StatementFieldRow
from .adapters.csmar_adapter import CsmarStatementAdapter
from .adapters.em_announce import EmNoticeAnnounceAdapter, build_csmar_announce_calendar
from .catalog import FACTOR_VERSION
from .compute import build_daily_factor_panel
from .coverage import CoverageFailed, assert_mvp_coverage, coverage_summary
from .ingest import (
    pit_years_for_trade_year,
    read_pit_rows,
    stage_from_adapters,
    write_pit_from_staging,
)
from .paths_local import default_factor_root, default_staging_root, local_factor_day_path, version_segment
from .publish import publish_day
from .qa import check_day
from .source_policy import COVERAGE_REPORT_MIN_DATE, DEFAULT_POLICY, FundCnSourcePolicy, StatementSource, resolve_policy
from .staging import (
    read_announce_rows,
    read_industry_map,
    read_share_capital_rows,
    read_statement_rows,
    write_industry_map,
    write_marker,
    write_share_capital_rows,
)
from .verify import VerifyFailed, run_free_source_verify

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
    report_start: date | None = None,
    report_end: date | None = None,
    policy: FundCnSourcePolicy | None = None,
    csmar_path: str | Path | None = None,
    skip_verify: bool = False,
) -> dict[str, Any]:
    """拉/注入财报、公告、股本并写入 staging。

    默认 ``policy.statement_source=CSMAR``：科目只从国泰安读入，公告用 EM NOTICE 补洞。
    ``start/end`` 兼容旧语义；优先用 ``report_start/report_end`` 过滤报告期。
    """
    cfg = policy or DEFAULT_POLICY
    stmts = list(statement_rows) if statement_rows is not None else []
    anns = list(announce_rows) if announce_rows is not None else []
    industry_rows: list[Any] = []
    adapter: Any = None
    # 报告期窗口：显式 report_* 优先，否则回退 start/end
    rs = report_start if report_start is not None else start
    re = report_end if report_end is not None else end

    if statement_rows is None:
        if statement_adapter is not None:
            adapter = statement_adapter
        elif cfg.statement_source == StatementSource.CSMAR:
            adapter = CsmarStatementAdapter(path=csmar_path)
        else:
            adapter = AkshareStatementAdapter()
        stmts = adapter.fetch_statements(instrument_keys, start=rs, end=re)
        if isinstance(adapter, CsmarStatementAdapter):
            industry_rows = adapter.fetch_industry_map()

    if announce_rows is None:
        if announce_adapter is not None:
            anns = announce_adapter.fetch_announcements(instrument_keys, start=rs, end=re)
        elif cfg.statement_source == StatementSource.CSMAR:
            corrects = []
            if isinstance(adapter, CsmarStatementAdapter):
                corrects = adapter.fetch_correct_announcements()
            anns = build_csmar_announce_calendar(
                instrument_keys=instrument_keys,
                csmar_correct_rows=corrects,
                em_adapter=EmNoticeAnnounceAdapter(),
                start=rs,
                end=re,
            )
        else:
            adapter_a = adapter if hasattr(adapter, "fetch_announcements") else StaticAnnounceAdapter([])
            anns = adapter_a.fetch_announcements(instrument_keys, start=rs, end=re)

    info = stage_from_adapters(stmts, anns, root=root)
    if industry_rows:
        write_industry_map(industry_rows, root=root)
        info["industry_rows"] = len(industry_rows)

    # 国泰安策略：默认不把巨潮事件写入主路径；仅显式注入或 use_cninfo_share_events 时写入
    shares: list[ShareCapitalRow] = list(share_capital_rows or [])
    if share_capital_rows is None and share_capital_adapter is not None and cfg.use_cninfo_share_events:
        shares = share_capital_adapter.fetch_share_capital(instrument_keys, start=rs, end=re)
    if share_capital_rows is not None or (share_capital_adapter is not None and cfg.use_cninfo_share_events):
        write_share_capital_rows(shares, root=root)
    info["share_capital_rows"] = len(shares)
    info["source_policy"] = {
        "statement_source": cfg.statement_source.value,
        "prefer_report_shares": cfg.prefer_report_shares,
        "use_cninfo_share_events": cfg.use_cninfo_share_events,
        "verify_with_free_sources": cfg.verify_with_free_sources,
    }

    verify_payload: dict[str, Any] | None = None
    if (
        not skip_verify
        and cfg.verify_with_free_sources
        and cfg.statement_source == StatementSource.CSMAR
        and statement_rows is None
    ):
        try:
            verify_payload = run_free_source_verify(
                csmar_statements=read_statement_rows(root),
                instrument_keys=instrument_keys,
                policy=cfg,
                root=root,
                start=rs,
                end=re,
                require=False,
            )
        except Exception as exc:
            _log.warning("free-source verify soft-failed at stage: %s", exc)
            verify_payload = {"ok": False, "summary": str(exc)}
    if verify_payload is not None:
        info["verify"] = verify_payload
    write_marker("source_policy", info.get("source_policy") or {}, root=root)
    return info


def run_coverage(
    root: Path | None = None,
    *,
    require: bool = False,
    verify_payload: dict[str, Any] | None = None,
    report_min: date | None = COVERAGE_REPORT_MIN_DATE,
) -> dict[str, Any]:
    """写 coverage marker；require=True 时 MVP 未过则抛 CoverageFailed。

    统计窗口默认报告期 ≥2005；缺公告行应在 PIT 前剔除，使挂载率可过 5% 硬闸。
    """
    from .ingest import prepare_statements_for_pit

    statements = read_statement_rows(root)
    announce = read_announce_rows(root)
    industry = read_industry_map(root)
    # Coverage 按「剔除后」口径统计，与即将进 PIT 的集合一致
    kept, missing, prep = prepare_statements_for_pit(
        statements, announce, report_min=report_min
    )
    write_marker(
        "missing_announce",
        {
            "count": int(len(missing)),
            "report_min": prep.get("report_min"),
            "keys": missing.head(5000).to_dict("records") if not missing.empty else [],
        },
        root=root,
    )
    summary = coverage_summary(
        kept,
        announce,
        verify_payload=verify_payload,
        industry_map=industry,
        report_min=report_min,
    )
    summary["prepare"] = prep
    write_marker("coverage", summary, root=root)
    if require:
        assert_mvp_coverage(summary)
        attach = summary.get("announce_attach") or {}
        if attach and not attach.get("ok", True):
            raise CoverageFailed(
                summary,
                [f"announce_attach_rate={attach.get('missing_announce_rate')}"],
            )
    return summary


def run_pit(
    root: Path | None = None,
    *,
    report_min: date | None = COVERAGE_REPORT_MIN_DATE,
) -> dict[str, Any]:
    """staging → 本地 pit/metrics.parquet + by_year 分片。"""
    return write_pit_from_staging(root, report_min=report_min)


def _day_market(market: pd.DataFrame | None, trading_date: str) -> pd.DataFrame | None:
    if market is None or market.empty:
        return market
    iso = str(trading_date)[:10] if "-" in str(trading_date) else (
        f"{str(trading_date)[:4]}-{str(trading_date)[4:6]}-{str(trading_date)[6:8]}"
    )
    frame = market.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.strftime("%Y-%m-%d")
    return frame[frame["trading_date"] == iso].copy()


def _year_factor_complete(
    factor_root: Path,
    year_days: Sequence[str],
    *,
    factor_version: int = FACTOR_VERSION,
) -> bool:
    """当年全部交易日宽表已落盘则视为可跳过（断点续跑）。"""
    if not year_days:
        return True
    for iso in year_days:
        path = local_factor_day_path(factor_root, iso, factor_version=factor_version)
        if not path.is_file():
            return False
    return True


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
    prefer_report_shares: bool = True,
    qlib_dir: str | Path | None = None,
    resume: bool = True,
    resume_from: str | int | None = None,
) -> dict[str, Any]:
    """按年算因子 → 质检 → 通过才写 parquet/R2。

    - PIT 按年分片加载（Y-2..Y），避免全量 千万行常驻。
    - asof 每年建一次；QA 抽样只对样本票重建，不再扫全量 PIT。
    - ``resume=True`` 时跳过已完整落盘的年份；``resume_from`` 可指定起始年。
    """
    from .market import load_unadjusted_close

    shares = read_share_capital_rows(root)
    industry = read_industry_map(root)
    froot = factor_root or default_factor_root()
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped_years: list[str] = []

    iso_days = [
        (d.isoformat() if hasattr(d, "isoformat") else str(d))[:10] for d in trading_dates
    ]
    by_year: dict[str, list[str]] = {}
    for iso in iso_days:
        by_year.setdefault(iso[:4], []).append(iso)

    resume_year = int(resume_from) if resume_from is not None else None

    for year, year_days in sorted(by_year.items()):
        y_int = int(year)
        if resume_year is not None and y_int < resume_year:
            skipped_years.append(year)
            print(f"[compute] year={year} skip before resume_from={resume_year}", flush=True)
            continue
        if resume and _year_factor_complete(froot, year_days):
            skipped_years.append(year)
            print(f"[compute] year={year} skip complete days={len(year_days)}", flush=True)
            continue

        # 仅载入当年可见所需 PIT 分片
        if pit_rows is not None:
            rows = list(pit_rows)
        else:
            rows = read_pit_rows(root, years=pit_years_for_trade_year(y_int))

        year_mkt = None
        if market is not None and not market.empty:
            m = market.copy()
            m["trading_date"] = pd.to_datetime(m["trading_date"]).dt.strftime("%Y-%m-%d")
            year_mkt = m[m["trading_date"].isin(year_days)].copy()
        elif qlib_dir is not None or market is None:
            try:
                year_mkt = load_unadjusted_close(
                    instrument_keys,
                    year_days,
                    qlib_dir=qlib_dir,
                    require_factor=True,
                )
            except Exception as exc:
                _log.warning("year=%s market load failed: %s", year, exc)
                year_mkt = pd.DataFrame(columns=["instrument_key", "trading_date", "close"])
        if year_mkt is not None and not year_mkt.empty:
            year_keys = sorted(year_mkt["instrument_key"].astype(str).unique())
        else:
            year_keys = list(instrument_keys)
        print(
            f"[compute] year={year} days={len(year_days)} symbols={len(year_keys)} pit_rows={len(rows)}",
            flush=True,
        )
        year_panel = build_daily_factor_panel(
            rows,
            year_days,
            year_keys,
            year_mkt,
            mvp_only=mvp_only,
            share_capital=shares,
            industry_map=industry,
            prefer_report_shares=prefer_report_shares,
        )
        for iso in year_days:
            # 已有单日文件且 resume：跳过该日（年未齐时的细粒度续跑）
            day_path = local_factor_day_path(froot, iso)
            if resume and day_path.is_file():
                published.append({"trading_date": iso, "path": str(day_path), "row_count": 0, "resumed": True})
                continue
            day_mkt = _day_market(year_mkt if year_mkt is not None else market, iso)
            if year_panel is None or year_panel.empty:
                panel = year_panel
            else:
                panel = year_panel[year_panel["trading_date"].astype(str).str[:10] == iso].copy()
            if (
                panel is not None
                and not panel.empty
                and day_mkt is not None
                and not day_mkt.empty
                and "instrument_key" in day_mkt.columns
            ):
                live = set(day_mkt["instrument_key"].astype(str))
                panel = panel[panel["instrument_key"].astype(str).isin(live)].copy()
            qa = check_day(
                trading_date=iso,
                wide=panel,
                pit_rows=rows,
                market=day_mkt,
                share_capital=shares,
                industry_map=industry,
                mvp_only=mvp_only,
                prefer_report_shares=prefer_report_shares,
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
            if len(published) % 50 == 0:
                print(f"[{iso}] published cumulative={len(published)}", flush=True)
        print(
            f"[compute] year={year} done published={len(published)} failed={len(failed)}",
            flush=True,
        )

    write_marker(
        "factor_publish",
        {
            "status": "done" if not failed else "partial",
            "days": len(published),
            "failed_days": [item["trading_date"] for item in failed],
            "skipped_years": skipped_years,
            "row_count": int(sum(int(item.get("row_count") or 0) for item in published)),
            "factor_version_dir": version_segment(),
        },
        root=root,
    )
    return {
        "days": published,
        "failed": failed,
        "skipped_years": skipped_years,
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
    policy: FundCnSourcePolicy | None = None,
    csmar_path: str | Path | None = None,
    skip_verify: bool = False,
    qlib_dir: str | Path | None = None,
    report_start: date | None = None,
    report_end: date | None = None,
    resume: bool = True,
    resume_from: str | int | None = None,
) -> dict[str, Any]:
    """离线可注入行；直播数走 adapter。coverage 默认硬闸。"""
    cfg = policy or DEFAULT_POLICY
    staging_root = root or default_staging_root()
    stage_info = run_stage(
        instrument_keys,
        statement_rows=statement_rows,
        announce_rows=announce_rows,
        share_capital_rows=share_capital_rows,
        share_capital_adapter=share_capital_adapter,
        root=staging_root,
        policy=cfg,
        csmar_path=csmar_path,
        skip_verify=skip_verify or statement_rows is not None,
        report_start=report_start,
        report_end=report_end,
    )
    verify_payload = stage_info.get("verify")
    coverage = run_coverage(
        staging_root,
        require=require_coverage,
        verify_payload=verify_payload if require_coverage and not skip_verify else None,
        report_min=cfg.coverage_report_min,
    )
    if (
        require_coverage
        and not skip_verify
        and statement_rows is None
        and cfg.verify_with_free_sources
        and verify_payload is not None
        and verify_payload.get("ok") is False
        and not verify_payload.get("skipped")
    ):
        raise VerifyFailed(verify_payload)

    pit = run_pit(staging_root, report_min=cfg.coverage_report_min)
    factors = run_compute_publish(
        trading_dates,
        instrument_keys,
        market,
        root=staging_root,
        factor_root=factor_root,
        mvp_only=mvp_only,
        upload=upload,
        qa_dir=qa_dir,
        prefer_report_shares=cfg.prefer_report_shares,
        qlib_dir=qlib_dir,
        resume=resume,
        resume_from=resume_from,
    )
    return {
        "stage": stage_info,
        "coverage": coverage,
        "pit": pit,
        "factors": factors,
        "policy": cfg.statement_source.value,
    }


__all__ = [
    "CoverageFailed",
    "VerifyFailed",
    "resolve_policy",
    "run_compute_publish",
    "run_coverage",
    "run_pipeline",
    "run_pit",
    "run_stage",
]
