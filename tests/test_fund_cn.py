# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fund_cn 公式 / PIT / coverage / 质检闸；不含 Factor Lab / D1。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from qlib.contrib.data.fund_cn.adapters.announce import as_utc
from qlib.contrib.data.fund_cn.adapters.base import (
    AnnounceRow,
    ShareCapitalRow,
    StatementFieldRow,
)
from qlib.contrib.data.fund_cn.catalog import (
    FACTOR_DEFINITIONS,
    FACTOR_SET,
    backtest_eligible_factors,
    coverage_matrix,
    effective_factor,
    mvp_factors,
    promote_factor_to_pit_safe,
    schema_hash,
)
from qlib.contrib.data.fund_cn.compute import (
    MarketRequiredError,
    build_daily_factor_panel,
    forward_fill_report_factors,
)
from qlib.contrib.data.fund_cn.coverage import (
    METRIC_TO_STATEMENT_FIELDS,
    CoverageFailed,
    announce_attach_rate,
    assert_mvp_coverage,
    coverage_summary,
    drop_statements_without_announce,
    evaluate_metric_coverage,
    filter_statements_by_report_min,
    resolve_statement_fields,
)
from qlib.contrib.data.fund_cn.ingest import (
    pit_years_for_trade_year,
    stage_from_adapters,
    write_pit_from_staging,
)
from qlib.contrib.data.fund_cn.market import instrument_to_qlib, qlib_to_instrument
from qlib.contrib.data.fund_cn.metrics import (
    attach_announce,
    compute_period_metrics,
    pivot_statements,
    statements_to_pit_metrics,
    to_single_quarter,
)
from qlib.contrib.data.fund_cn.pipeline import run_compute_publish, run_pipeline
from qlib.contrib.data.fund_cn.qa import check_day
from qlib.contrib.data.fund_cn.source_policy import COVERAGE_REPORT_MIN_DATE, RESTATEMENT_UNRECOVERABLE
from qlib.contrib.data.fund_cn.staging import read_announce_rows, read_statement_rows


def _stmt(
    ik: str,
    period: date,
    field: str,
    value: float,
    *,
    statement_type: str = "income",
    q: int | None = None,
    revision: int = 0,
) -> StatementFieldRow:
    quarter = q if q is not None else (period.month - 1) // 3 + 1
    return StatementFieldRow(
        instrument_key=ik,
        report_period_end=period,
        report_period_start=None,
        fiscal_year=period.year,
        fiscal_quarter=quarter,
        statement_type=statement_type,
        field_code=field,
        value=value,
        source="test",
        source_record_id=f"{ik}:{period}:{field}:r{revision}",
        revision=revision,
    )


def _ann(ik: str, period: date, available: datetime, revision: int = 0) -> AnnounceRow:
    return AnnounceRow(
        instrument_key=ik,
        report_period_end=period,
        publish_time=available,
        available_time=available,
        revision=revision,
        is_restatement=revision > 0,
        source="test",
        source_record_id=f"{ik}:{period}:r{revision}",
    )


@pytest.fixture
def sample_universe() -> str:
    return "CNStock:000001"


def test_catalog_has_thirty_factors_and_mvp_subset():
    assert FACTOR_SET == "fund_cn_basic@v1"
    assert len(FACTOR_DEFINITIONS) == 30
    mvp = mvp_factors()
    assert len(mvp) >= 12
    assert all(f.enabled and f.information_policy == "PIT_SAFE" for f in mvp)
    assert schema_hash()
    matrix = coverage_matrix()
    assert len(matrix) == 30


def test_revision_snapshot_no_lookahead(sample_universe: str):
    """revision=0 保留首发值，后续修订不污染。"""
    ik = sample_universe
    pe = date(2023, 12, 31)
    t0 = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    t1 = datetime(2024, 4, 15, 10, 0, tzinfo=timezone.utc)
    statements = [
        _stmt(ik, pe, "net_profit_parent", 100.0, revision=0),
        _stmt(ik, pe, "shareholder_equity", 1000.0, statement_type="balance", revision=0),
        _stmt(ik, pe, "net_profit_parent", 120.0, revision=1),
        _stmt(ik, pe, "shareholder_equity", 1000.0, statement_type="balance", revision=1),
    ]
    announces = [_ann(ik, pe, t0, revision=0), _ann(ik, pe, t1, revision=1)]
    rows = statements_to_pit_metrics(
        pd.DataFrame([s.__dict__ for s in statements]),
        pd.DataFrame([a.__dict__ for a in announces]),
    )
    roe_rows = [r for r in rows if r["metric_code"] == "roe"]
    by_rev = {r["revision"]: r for r in roe_rows}
    assert by_rev[0]["value"] == pytest.approx(0.1)
    assert by_rev[1]["value"] == pytest.approx(0.12)


def test_to_single_quarter_and_ttm(sample_universe: str):
    ik = sample_universe
    periods = [
        (date(2023, 3, 31), 1, 10.0),
        (date(2023, 6, 30), 2, 25.0),
        (date(2023, 9, 30), 3, 45.0),
        (date(2023, 12, 31), 4, 100.0),
    ]
    statements = []
    announces = []
    for pe, q, cum in periods:
        statements.append(_stmt(ik, pe, "revenue", cum, q=q))
        statements.append(_stmt(ik, pe, "net_profit_parent", cum / 10, q=q))
        statements.append(_stmt(ik, pe, "shareholder_equity", 200.0, statement_type="balance", q=q))
        announces.append(_ann(ik, pe, datetime(2023, pe.month, 28, 15, 0, tzinfo=timezone.utc)))
    wide = pivot_statements(pd.DataFrame([s.__dict__ for s in statements]))
    joined = attach_announce(wide, pd.DataFrame([a.__dict__ for a in announces]))
    sq = to_single_quarter(joined)
    fy = sq[sq["report_period_end"] == date(2023, 12, 31)].iloc[0]
    assert fy["revenue__sq"] == pytest.approx(55.0)
    ttm = [
        r
        for r in compute_period_metrics(joined)
        if r["metric_code"] == "revenue_ttm" and r["report_period_end"] == date(2023, 12, 31)
    ]
    assert ttm and ttm[0]["value"] == pytest.approx(100.0)


def test_coverage_requires_announce(tmp_path: Path, sample_universe: str):
    ik = sample_universe
    statements = [
        _stmt(ik, date(2023, 12, 31), "revenue", 100.0),
        _stmt(ik, date(2023, 12, 31), "net_profit_parent", 10.0),
        _stmt(ik, date(2023, 12, 31), "shareholder_equity", 100.0, statement_type="balance"),
        _stmt(ik, date(2023, 12, 31), "total_assets", 200.0, statement_type="balance"),
        _stmt(ik, date(2023, 12, 31), "total_liabilities", 100.0, statement_type="balance"),
        _stmt(ik, date(2023, 12, 31), "gross_profit", 40.0),
        _stmt(ik, date(2023, 12, 31), "cash_from_operations", 12.0, statement_type="cashflow"),
        _stmt(ik, date(2023, 12, 31), "shares_outstanding", 10.0, statement_type="balance"),
    ]
    stage_from_adapters(statements, [], root=tmp_path)
    summary = coverage_summary(read_statement_rows(tmp_path), read_announce_rows(tmp_path))
    assert summary["eligible_count"] == 0
    with pytest.raises(CoverageFailed):
        assert_mvp_coverage(summary)


def test_coverage_announcement_must_match_revision(sample_universe: str):
    ik = sample_universe
    pe = date(2023, 12, 31)
    statements = pd.DataFrame(
        [
            _stmt(ik, pe, "net_profit_parent", 10.0, revision=0).__dict__,
            _stmt(ik, pe, "net_profit_parent", 12.0, revision=1).__dict__,
        ]
    )
    announce = pd.DataFrame(
        [_ann(ik, pe, datetime(2024, 3, 28, 15, tzinfo=timezone.utc), revision=0).__dict__]
    )
    report = evaluate_metric_coverage(statements, announce, "net_profit_parent", max_missing_announce=0.0)
    assert report.missing_announce_rate == pytest.approx(0.5)
    assert not report.pit_ok


def test_valuation_requires_market(sample_universe: str):
    ik = sample_universe
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    pit_rows = [
        {
            "instrument_key": ik,
            "metric_code": "roe",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 0.1,
        }
    ]
    with pytest.raises(MarketRequiredError):
        build_daily_factor_panel(pit_rows, [date(2024, 4, 1)], [ik], market=None, mvp_only=True)


def test_valuation_panel_uses_market_and_shares(sample_universe: str):
    ik = sample_universe
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    pit_rows = [
        {"instrument_key": ik, "metric_code": "shares_outstanding", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 10.0},
        {"instrument_key": ik, "metric_code": "shareholder_equity", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 100.0},
        {"instrument_key": ik, "metric_code": "net_income_ttm", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 20.0},
        {"instrument_key": ik, "metric_code": "revenue_ttm", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 200.0},
        {"instrument_key": ik, "metric_code": "cash_from_operations_ttm", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 25.0},
        {"instrument_key": ik, "metric_code": "roe", "report_period_end": date(2023, 12, 31), "available_time": available, "revision": 0, "value": 0.1},
    ]
    market = pd.DataFrame([{"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 10.0}])
    panel = build_daily_factor_panel(pit_rows, [date(2024, 4, 1)], [ik], market, mvp_only=True)
    assert abs(float(panel.iloc[0]["pb"]) - 1.0) < 1e-6
    assert abs(float(panel.iloc[0]["pe_ttm"]) - 5.0) < 1e-6
    assert abs(float(panel.iloc[0]["roe"]) - 0.1) < 1e-6


def test_metric_deps_all_mapped():
    deps = {d for f in FACTOR_DEFINITIONS for d in f.metric_deps}
    for dep in deps:
        assert resolve_statement_fields(dep)
        assert dep in METRIC_TO_STATEMENT_FIELDS


def test_promote_enables_backtest_eligibility():
    pending = next(f for f in FACTOR_DEFINITIONS if not f.mvp and f.factor_id == "operating_margin")
    before = {f.factor_id for f in backtest_eligible_factors()}
    assert pending.factor_id not in before
    updated = promote_factor_to_pit_safe(pending.factor_id)
    assert updated.all_gates_ok
    assert effective_factor(pending.factor_id).enabled is True
    assert pending.factor_id in {f.factor_id for f in backtest_eligible_factors()}


def test_date_only_announce_visible_next_day(sample_universe: str):
    ik = sample_universe
    day = date(2024, 3, 28)
    at = as_utc(day)
    assert at == datetime(2024, 3, 28, 16, 0, tzinfo=timezone.utc)
    rows = [
        {
            "instrument_key": ik,
            "metric_code": "roe",
            "report_period_end": date(2023, 12, 31),
            "available_time": at,
            "revision": 0,
            "value": 0.1,
        }
    ]
    panel = forward_fill_report_factors(rows, [day, date(2024, 3, 29)], [ik], factor_ids_subset=["roe"])
    by_day = dict(zip(panel["trading_date"], panel["roe"]))
    assert by_day["2024-03-28"] is None or pd.isna(by_day["2024-03-28"])
    assert by_day["2024-03-29"] == pytest.approx(0.1)


def test_instrument_code_roundtrip():
    assert instrument_to_qlib("CNStock:600000") == "SH600000"
    assert instrument_to_qlib("CNStock:000001") == "SZ000001"
    assert qlib_to_instrument("SH600000") == "CNStock:600000"


def test_unadjusted_close_is_close_over_factor():
    """估值必须用 raw_close = 复权 close / factor。"""
    adj_close = 12.0
    factor = 1.2
    raw = adj_close / factor
    assert raw == pytest.approx(10.0)
    market = pd.DataFrame(
        [{"instrument_key": "CNStock:000001", "trading_date": date(2024, 4, 1), "close": raw}]
    )
    pit_rows = [
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "shares_outstanding",
            "report_period_end": date(2023, 12, 31),
            "available_time": datetime(2024, 3, 28, 15, tzinfo=timezone.utc),
            "revision": 0,
            "value": 10.0,
        },
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "net_income_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": datetime(2024, 3, 28, 15, tzinfo=timezone.utc),
            "revision": 0,
            "value": 20.0,
        },
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "shareholder_equity",
            "report_period_end": date(2023, 12, 31),
            "available_time": datetime(2024, 3, 28, 15, tzinfo=timezone.utc),
            "revision": 0,
            "value": 100.0,
        },
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "revenue_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": datetime(2024, 3, 28, 15, tzinfo=timezone.utc),
            "revision": 0,
            "value": 200.0,
        },
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "roe",
            "report_period_end": date(2023, 12, 31),
            "available_time": datetime(2024, 3, 28, 15, tzinfo=timezone.utc),
            "revision": 0,
            "value": 0.2,
        },
    ]
    panel = build_daily_factor_panel(pit_rows, [date(2024, 4, 1)], ["CNStock:000001"], market, mvp_only=True)
    # mcap = 10 * 10 = 100；PE = 100 / 20 = 5
    assert float(panel.iloc[0]["pe_ttm"]) == pytest.approx(5.0)


def _full_pack(ik: str, pe: date, q: int, rev: int, revenue_cum: float, np_cum: float, equity: float):
    return [
        _stmt(ik, pe, "revenue", revenue_cum, q=q, revision=rev),
        _stmt(ik, pe, "net_profit_parent", np_cum, q=q, revision=rev),
        _stmt(ik, pe, "gross_profit", revenue_cum * 0.4, q=q, revision=rev),
        _stmt(ik, pe, "total_assets", equity * 2, statement_type="balance", q=q, revision=rev),
        _stmt(ik, pe, "total_liabilities", equity, statement_type="balance", q=q, revision=rev),
        _stmt(ik, pe, "shareholder_equity", equity, statement_type="balance", q=q, revision=rev),
        _stmt(ik, pe, "cash_from_operations", np_cum * 1.1, statement_type="cashflow", q=q, revision=rev),
        _stmt(ik, pe, "shares_outstanding", 10.0, statement_type="balance", q=q, revision=rev),
    ]


def test_pipeline_writes_local_parquet_without_registry(tmp_path: Path, sample_universe: str):
    """注入行跑通后应有本地日 parquet，且不依赖 Canonical/Registry。"""
    ik = sample_universe
    p2023 = date(2023, 12, 31)
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    restated = datetime(2024, 4, 15, 10, 0, tzinfo=timezone.utc)
    statements = []
    statements += _full_pack(ik, date(2022, 12, 31), 4, 0, 80.0, 8.0, 100.0)
    statements += _full_pack(ik, date(2023, 3, 31), 1, 0, 20.0, 2.0, 105.0)
    statements += _full_pack(ik, date(2023, 6, 30), 2, 0, 45.0, 5.0, 107.0)
    statements += _full_pack(ik, date(2023, 9, 30), 3, 0, 75.0, 7.5, 108.0)
    statements += _full_pack(ik, p2023, 4, 0, 100.0, 10.0, 110.0)
    statements += _full_pack(ik, p2023, 4, 1, 110.0, 12.0, 110.0)
    announces = [
        _ann(ik, date(2022, 12, 31), datetime(2023, 3, 30, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 3, 31), datetime(2023, 4, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 6, 30), datetime(2023, 8, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 9, 30), datetime(2023, 10, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, p2023, available, revision=0),
        _ann(ik, p2023, restated, revision=1),
    ]
    market = pd.DataFrame(
        [
            {"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 11.0},
            {"instrument_key": ik, "trading_date": date(2024, 4, 16), "close": 12.0},
        ]
    )
    result = run_pipeline(
        [ik],
        [date(2024, 4, 1), date(2024, 4, 16)],
        statement_rows=statements,
        announce_rows=announces,
        share_capital_rows=[
            ShareCapitalRow(ik, date(2023, 12, 31), datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc), 10.0, source="test")
        ],
        market=market,
        root=tmp_path / "staging",
        factor_root=tmp_path / "factors",
        mvp_only=True,
        require_coverage=True,
        upload=False,
        qa_dir=tmp_path / "qa",
    )
    assert result["pit"]["row_count"] > 0
    assert result["factors"]["row_count"] > 0
    assert "registered" not in result
    days = {item["trading_date"]: item for item in result["factors"]["days"]}
    assert Path(days["2024-04-01"]["path"]).is_file()
    panel = pd.read_parquet(days["2024-04-01"]["path"])
    assert float(panel.iloc[0]["pe_ttm"]) == pytest.approx(11.0)


def test_coverage_fail_does_not_write_panel(tmp_path: Path, sample_universe: str):
    """无公告时 coverage 硬闸，不写 PIT / 日宽表。"""
    ik = sample_universe
    statements = [_stmt(ik, date(2023, 12, 31), "revenue", 100.0)]
    with pytest.raises(CoverageFailed):
        run_pipeline(
            [ik],
            [date(2024, 4, 1)],
            statement_rows=statements,
            announce_rows=[],
            market=pd.DataFrame([{"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 10.0}]),
            root=tmp_path / "staging",
            factor_root=tmp_path / "factors",
            require_coverage=True,
            upload=False,
            qa_dir=tmp_path / "qa",
        )
    assert not list((tmp_path / "factors").rglob("*.parquet"))


def test_day_qa_all_nan_skips_publish(tmp_path: Path):
    """交易日早于 available_time 时全日空值，不写 parquet。"""
    later = datetime(2024, 5, 1, 15, 0, tzinfo=timezone.utc)
    pit_rows = [
        {
            "instrument_key": "CNStock:000001",
            "metric_code": "roe",
            "report_period_end": date(2023, 12, 31),
            "available_time": later,
            "revision": 0,
            "value": 0.1,
        }
    ]
    market = pd.DataFrame(
        [{"instrument_key": "CNStock:000001", "trading_date": date(2024, 4, 1), "close": 10.0}]
    )
    result = run_compute_publish(
        [date(2024, 4, 1)],
        ["CNStock:000001"],
        market,
        pit_rows=pit_rows,
        root=tmp_path / "staging",
        factor_root=tmp_path / "factors",
        upload=False,
        qa_dir=tmp_path / "qa",
    )
    assert result["days"] == []
    assert result["failed"]
    assert not list((tmp_path / "factors").rglob("????????.parquet"))


def test_check_day_empty_fails(tmp_path: Path):
    payload = check_day(
        trading_date="2024-04-01",
        wide=pd.DataFrame(),
        pit_rows=[],
        market=None,
        qa_dir=tmp_path,
    )
    assert payload["ok"] is False
    assert any("空" in e for e in payload["errors"])


def test_instrument_to_qlib_beijing():
    assert instrument_to_qlib("CNStock:430047") == "BJ430047"
    assert instrument_to_qlib("CNStock:920001") == "BJ920001"
    assert qlib_to_instrument("BJ430047") == "CNStock:430047"


def test_csmar_dataframe_to_rows_filters_typrep_and_maps_fields():
    from qlib.contrib.data.fund_cn.adapters.csmar_adapter import dataframe_to_csmar_rows
    from qlib.contrib.data.fund_cn.source_policy import is_financial_industry

    frame = pd.DataFrame(
        [
            {
                "证券代码": "600000",
                "季度日期": "2023-12-31",
                "报表类型编码": "A",
                "是否发生差错更正": "",
                "差错更正披露日期": "",
                "行业代码": "J66",
                "行业名称": "货币金融服务",
                "营业收入": 100.0,
                "营业成本": 40.0,
                "归属于母公司所有者的净利润": 10.0,
                "经营活动产生的现金流量净额": 12.0,
                "资产总计": 200.0,
                "负债合计": 90.0,
                "归属于母公司所有者权益合计": 110.0,
                "实收资本(或股本)": 50.0,
            },
            {
                "证券代码": "600000",
                "季度日期": "2023-12-31",
                "报表类型编码": "B",  # 母公司，应过滤
                "营业收入": 999.0,
                "归属于母公司所有者的净利润": 99.0,
                "资产总计": 1.0,
                "负债合计": 1.0,
                "归属于母公司所有者权益合计": 1.0,
                "经营活动产生的现金流量净额": 1.0,
                "实收资本(或股本)": 1.0,
            },
        ]
    )
    stmts, corrects, industries = dataframe_to_csmar_rows(frame, instrument_keys=["CNStock:600000"])
    assert corrects == []
    assert any(r.field_code == "revenue" and r.value == 100.0 for r in stmts)
    assert all(r.source == "csmar" for r in stmts)
    assert not any(r.value == 999.0 for r in stmts)
    assert industries and is_financial_industry(industries[0].industry_code)


def test_csmar_correct_announce_revision():
    from qlib.contrib.data.fund_cn.adapters.csmar_adapter import dataframe_to_csmar_rows

    frame = pd.DataFrame(
        [
            {
                "证券代码": "000001",
                "季度日期": "2023-12-31",
                "报表类型编码": "A",
                "是否发生差错更正": "1",
                "差错更正披露日期": "2024-04-15",
                "行业代码": "J66",
                "营业收入": 100.0,
                "归属于母公司所有者的净利润": 10.0,
                "资产总计": 200.0,
                "负债合计": 100.0,
                "归属于母公司所有者权益合计": 100.0,
                "经营活动产生的现金流量净额": 11.0,
                "实收资本(或股本)": 10.0,
            }
        ]
    )
    stmts, corrects, _ = dataframe_to_csmar_rows(frame)
    assert all(r.revision == 1 for r in stmts)
    assert len(corrects) == 1
    assert corrects[0].revision == 1
    assert corrects[0].source == "csmar_correct_declare"


def test_compare_statement_samples_mismatch_rate():
    from qlib.contrib.data.fund_cn.verify import compare_statement_samples

    pe = date(2023, 12, 31)
    ik = "CNStock:600000"
    csmar = [
        _stmt(ik, pe, "revenue", 100.0),
        _stmt(ik, pe, "net_profit_parent", 10.0),
    ]
    free_ok = [
        _stmt(ik, pe, "revenue", 100.5),
        _stmt(ik, pe, "net_profit_parent", 10.1),
    ]
    free_bad = [
        _stmt(ik, pe, "revenue", 200.0),
        _stmt(ik, pe, "net_profit_parent", 10.0),
    ]
    ok = compare_statement_samples(csmar, free_ok, max_rel_error=0.05)
    assert ok["compared"] == 2
    assert ok["mismatch"] == 0
    bad = compare_statement_samples(csmar, free_bad, max_rel_error=0.05)
    assert bad["mismatch"] >= 1


def test_prefer_report_shares_and_shares_source_tag(sample_universe: str):
    from qlib.contrib.data.fund_cn.compute import compute_valuation_factors

    ik = sample_universe
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    pit_rows = [
        {
            "instrument_key": ik,
            "metric_code": "shares_outstanding",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 10.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "net_income_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 20.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "shareholder_equity",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 100.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "revenue_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 200.0,
        },
    ]
    market = pd.DataFrame([{"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 10.0}])
    # 事件股本故意不同：优先报告股本时 PE 仍按 10 股
    share_capital = pd.DataFrame(
        [
            {
                "instrument_key": ik,
                "effective_date": date(2024, 1, 1),
                "announce_time": available,
                "total_shares": 20.0,
                "source": "cninfo",
                "source_record_id": "x",
            }
        ]
    )
    panel = compute_valuation_factors(
        pit_rows,
        market,
        share_capital=share_capital,
        prefer_report_shares=True,
        mvp_only=True,
    )
    assert float(panel.iloc[0]["pe_ttm"]) == pytest.approx(5.0)
    assert panel.iloc[0]["shares_source"] == "report"


def test_non_financial_mask_clears_gross_margin_for_banks():
    from qlib.contrib.data.fund_cn.compute import apply_non_financial_mask

    panel = pd.DataFrame(
        [
            {"trading_date": "2024-04-01", "instrument_key": "CNStock:600000", "gross_margin": 0.3, "roe": 0.1},
            {"trading_date": "2024-04-01", "instrument_key": "CNStock:600519", "gross_margin": 0.5, "roe": 0.2},
        ]
    )
    industry = pd.DataFrame(
        [
            {"instrument_key": "CNStock:600000", "industry_code": "J66", "industry_name": "银行"},
            {"instrument_key": "CNStock:600519", "industry_code": "C15", "industry_name": "酒"},
        ]
    )
    out = apply_non_financial_mask(panel, industry, factor_ids_subset=["gross_margin", "roe"])
    bank = out[out["instrument_key"] == "CNStock:600000"].iloc[0]
    liquor = out[out["instrument_key"] == "CNStock:600519"].iloc[0]
    assert pd.isna(bank["gross_margin"])
    assert bank["roe"] == pytest.approx(0.1)
    assert liquor["gross_margin"] == pytest.approx(0.5)


def test_resolve_policy_defaults_to_csmar():
    from qlib.contrib.data.fund_cn.source_policy import StatementSource, resolve_policy

    policy = resolve_policy(source="csmar")
    assert policy.statement_source == StatementSource.CSMAR
    assert policy.prefer_report_shares is True
    assert policy.verify_with_free_sources is True


def test_csmar_adapter_reads_fixture_csv(tmp_path: Path):
    from qlib.contrib.data.fund_cn.adapters.csmar_adapter import CsmarStatementAdapter

    csv_path = tmp_path / "quarterly.csv"
    pd.DataFrame(
        [
            {
                "证券代码": "600519",
                "季度日期": "2023-12-31",
                "报表类型编码": "A",
                "是否发生差错更正": "",
                "差错更正披露日期": "",
                "行业代码": "C15",
                "行业名称": "酒、饮料和精制茶制造业",
                "营业收入": 1000.0,
                "营业成本": 300.0,
                "归属于母公司所有者的净利润": 400.0,
                "经营活动产生的现金流量净额": 450.0,
                "资产总计": 2000.0,
                "负债合计": 500.0,
                "归属于母公司所有者权益合计": 1500.0,
                "实收资本(或股本)": 100.0,
            }
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    adapter = CsmarStatementAdapter(path=csv_path, chunksize=10)
    rows = adapter.fetch_statements(["CNStock:600519"], start=date(2023, 1, 1), end=date(2023, 12, 31))
    assert any(r.field_code == "revenue" and r.value == 1000.0 for r in rows)
    assert adapter.fetch_industry_map()[0].industry_code == "C15"


def test_pipeline_with_industry_mask(tmp_path: Path):
    """注入 CSMAR 风格行业映射后，银行 gross_margin 应为空。"""
    from qlib.contrib.data.fund_cn.staging import write_industry_map

    ik = "CNStock:600000"
    p2023 = date(2023, 12, 31)
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    statements = []
    statements += _full_pack(ik, date(2022, 12, 31), 4, 0, 80.0, 8.0, 100.0)
    statements += _full_pack(ik, date(2023, 3, 31), 1, 0, 20.0, 2.0, 105.0)
    statements += _full_pack(ik, date(2023, 6, 30), 2, 0, 45.0, 5.0, 107.0)
    statements += _full_pack(ik, date(2023, 9, 30), 3, 0, 75.0, 7.5, 108.0)
    statements += _full_pack(ik, p2023, 4, 0, 100.0, 10.0, 110.0)
    announces = [
        _ann(ik, date(2022, 12, 31), datetime(2023, 3, 30, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 3, 31), datetime(2023, 4, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 6, 30), datetime(2023, 8, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, date(2023, 9, 30), datetime(2023, 10, 28, 15, 0, tzinfo=timezone.utc)),
        _ann(ik, p2023, available, revision=0),
    ]
    market = pd.DataFrame([{"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 11.0}])
    staging = tmp_path / "staging"
    write_industry_map(
        [{"instrument_key": ik, "industry_code": "J66", "industry_name": "银行", "as_of": p2023}],
        root=staging,
    )
    result = run_pipeline(
        [ik],
        [date(2024, 4, 1)],
        statement_rows=statements,
        announce_rows=announces,
        market=market,
        root=staging,
        factor_root=tmp_path / "factors",
        mvp_only=True,
        require_coverage=True,
        upload=False,
        qa_dir=tmp_path / "qa",
        skip_verify=True,
    )
    assert result["factors"]["row_count"] > 0
    panel = pd.read_parquet(result["factors"]["days"][0]["path"])
    assert pd.isna(panel.iloc[0]["gross_margin"])
    assert float(panel.iloc[0]["roe"]) == pytest.approx(panel.iloc[0]["roe"])


def test_drop_statements_without_announce_and_window():
    """缺公告修订行剔除后挂载率归零；窗口过滤掉早期报告期。"""
    ik = "CNStock:000001"
    early = date(1998, 12, 31)
    late = date(2023, 12, 31)
    statements = pd.DataFrame(
        [
            _stmt(ik, early, "revenue", 1.0).__dict__,
            _stmt(ik, late, "revenue", 100.0, revision=0).__dict__,
            _stmt(ik, late, "revenue", 110.0, revision=1).__dict__,
        ]
    )
    announce = pd.DataFrame(
        [_ann(ik, late, datetime(2024, 3, 28, 15, tzinfo=timezone.utc), revision=0).__dict__]
    )
    windowed = filter_statements_by_report_min(statements, report_min=COVERAGE_REPORT_MIN_DATE)
    assert early not in set(windowed["report_period_end"].tolist())
    kept, missing = drop_statements_without_announce(windowed, announce)
    assert len(missing) == 1
    assert int(missing.iloc[0]["revision"]) == 1
    assert set(kept["revision"].astype(int)) == {0}
    attach = announce_attach_rate(kept, announce, report_min=COVERAGE_REPORT_MIN_DATE)
    assert attach["missing_announce_rate"] == pytest.approx(0.0)
    assert attach["ok"]


def test_year_batch_panel_matches_single_day(sample_universe: str):
    """年批面板切片与单日单票重算应一致（QA 硬闸前提）。"""
    ik = sample_universe
    available = datetime(2024, 3, 28, 15, 0, tzinfo=timezone.utc)
    pit_rows = [
        {
            "instrument_key": ik,
            "metric_code": "debt_to_asset",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 0.5,
        },
        {
            "instrument_key": ik,
            "metric_code": "roe",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 0.12,
        },
        {
            "instrument_key": ik,
            "metric_code": "shares_outstanding",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 10.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "shareholder_equity",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 100.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "net_income_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 20.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "revenue_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 200.0,
        },
        {
            "instrument_key": ik,
            "metric_code": "cash_from_operations_ttm",
            "report_period_end": date(2023, 12, 31),
            "available_time": available,
            "revision": 0,
            "value": 25.0,
        },
    ]
    days = [date(2024, 4, 1), date(2024, 4, 2)]
    market = pd.DataFrame(
        [
            {"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 10.0},
            {"instrument_key": ik, "trading_date": date(2024, 4, 2), "close": 11.0},
        ]
    )
    year_panel = build_daily_factor_panel(pit_rows, days, [ik], market, mvp_only=True)
    day_panel = build_daily_factor_panel(
        pit_rows, [date(2024, 4, 1)], [ik], market[market["trading_date"] == date(2024, 4, 1)], mvp_only=True
    )
    year_row = year_panel[year_panel["trading_date"].astype(str).str[:10] == "2024-04-01"].iloc[0]
    for field in ("debt_to_asset", "roe", "pe_ttm", "pb"):
        assert year_row[field] == pytest.approx(float(day_panel.iloc[0][field]), rel=1e-5, abs=1e-8)


def test_resume_skips_complete_year(tmp_path: Path, sample_universe: str):
    """已完整落盘的年份在 resume 下应跳过。"""
    from qlib.contrib.data.fund_cn.storage import write_day

    ik = sample_universe
    factor_root = tmp_path / "factors"
    # 预先写入该交易日宽表，模拟上年已跑完
    write_day(
        factor_root,
        "2024-04-01",
        pd.DataFrame([{"trading_date": "2024-04-01", "instrument_key": ik, "roe": 0.1}]),
        mvp_only=True,
    )
    market = pd.DataFrame([{"instrument_key": ik, "trading_date": date(2024, 4, 1), "close": 10.0}])
    result = run_compute_publish(
        [date(2024, 4, 1)],
        [ik],
        market,
        pit_rows=[],
        root=tmp_path / "staging",
        factor_root=factor_root,
        upload=False,
        qa_dir=tmp_path / "qa",
        resume=True,
    )
    assert "2024" in result["skipped_years"]
    assert pit_years_for_trade_year(2000) == [1998, 1999, 2000]


def test_pit_shards_and_quality_tag(tmp_path: Path, sample_universe: str):
    """PIT 写 by_year 分片；更正行带 restatement_unrecoverable 标签。"""
    ik = sample_universe
    pe = date(2023, 12, 31)
    statements = [
        _stmt(ik, pe, "revenue", 100.0, revision=0),
        _stmt(ik, pe, "net_profit_parent", 10.0, revision=0),
        _stmt(ik, pe, "shareholder_equity", 100.0, statement_type="balance", revision=0),
        _stmt(ik, pe, "total_assets", 200.0, statement_type="balance", revision=0),
        _stmt(ik, pe, "total_liabilities", 100.0, statement_type="balance", revision=0),
        _stmt(ik, pe, "gross_profit", 40.0, revision=0),
        _stmt(ik, pe, "cash_from_operations", 12.0, statement_type="cashflow", revision=0),
        _stmt(ik, pe, "shares_outstanding", 10.0, statement_type="balance", revision=0),
        _stmt(ik, pe, "revenue", 110.0, revision=1),
        _stmt(ik, pe, "net_profit_parent", 12.0, revision=1),
        _stmt(ik, pe, "shareholder_equity", 100.0, statement_type="balance", revision=1),
        _stmt(ik, pe, "total_assets", 200.0, statement_type="balance", revision=1),
        _stmt(ik, pe, "total_liabilities", 100.0, statement_type="balance", revision=1),
        _stmt(ik, pe, "gross_profit", 44.0, revision=1),
        _stmt(ik, pe, "cash_from_operations", 13.0, statement_type="cashflow", revision=1),
        _stmt(ik, pe, "shares_outstanding", 10.0, statement_type="balance", revision=1),
    ]
    announces = [
        _ann(ik, pe, datetime(2024, 3, 28, 15, tzinfo=timezone.utc), revision=0),
        _ann(ik, pe, datetime(2024, 4, 15, 10, tzinfo=timezone.utc), revision=1),
    ]
    stage_from_adapters(statements, announces, root=tmp_path)
    info = write_pit_from_staging(tmp_path)
    assert info["row_count"] > 0
    assert "2024" in info["year_shards"]
    assert (tmp_path / "pit" / "by_year" / "2024.parquet").is_file()
    frame = pd.read_parquet(tmp_path / "pit" / "metrics.parquet")
    assert "quality_tag" in frame.columns
    restated = frame[frame["revision"] == 1]
    assert (restated["quality_tag"] == RESTATEMENT_UNRECOVERABLE).all()
