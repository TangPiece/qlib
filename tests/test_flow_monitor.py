# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""flow_monitor：价格调整公式、证据等级、对齐与矩阵；不打外网。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qlib.contrib.data.flow_monitor.adapters.base import FlowRow, row_from_def
from qlib.contrib.data.flow_monitor.catalog import (
    SERIES_DEFINITIONS,
    catalog_rows,
    series_by_id,
)
from qlib.contrib.data.flow_monitor.compute import (
    build_allocation_table,
    build_change_table,
    build_flow_vs_price,
    write_compute_outputs,
)
from qlib.contrib.data.flow_monitor.diagnostics import (
    allocation_evidence_matrix,
    classify_price_vs_flow,
    price_adjusted_flow,
    share_to_flow_value,
)
from qlib.contrib.data.flow_monitor.pipeline import run_pipeline
from qlib.contrib.data.flow_monitor.report import (
    question_is_real,
    question_source_dest,
    unavailable_series,
)
from qlib.contrib.data.flow_monitor.staging import read_long_rows, write_long_rows


def _row(series_id: str, day: date, value: float) -> FlowRow:
    return row_from_def(series_by_id()[series_id], day, value)


def test_catalog_has_evidence_grades_and_disabled_ici() -> None:
    grades = {row.evidence_grade for row in SERIES_DEFINITIONS}
    assert grades == {"L1", "L2", "L3"}
    lookup = series_by_id()
    assert lookup["gold_gld_change"].evidence_grade == "L1"
    assert lookup["px_spx"].evidence_grade == "L3"
    assert lookup["hsgt_north_hold_mcap"].evidence_grade == "L2"
    assert lookup["ici_mmf_aum"].enabled is False
    assert lookup["px_dxy"].enabled is False
    assert "EPFR" not in str(catalog_rows())
    # 北向必须带口径变更免责
    assert "披露" in lookup["hsgt_north_net"].notes


def test_price_adjusted_flow_subtracts_mark_to_market() -> None:
    # 市值 100→120，价格 +20%，真实净流入应为 0
    aum = pd.Series([100.0, 120.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    price = pd.Series([10.0, 12.0], index=aum.index)
    flow = price_adjusted_flow(aum, price)
    assert flow.iloc[1] == pytest.approx(0.0)
    # 市值 100→130，价格 +20%，估计流入 10
    aum2 = pd.Series([100.0, 130.0], index=aum.index)
    flow2 = price_adjusted_flow(aum2, price)
    assert flow2.iloc[1] == pytest.approx(10.0)


def test_share_change_times_price_is_not_price_return() -> None:
    shares = pd.Series([10.0, 11.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    price = pd.Series([2.0, 2.0], index=shares.index)
    flow = share_to_flow_value(shares, price)
    assert flow.iloc[1] == pytest.approx(2.0)


def test_classify_does_not_treat_price_as_inflow() -> None:
    assert classify_price_vs_flow(price_change=0.12, flow=-0.01, has_l1=True) == "PRICE_DRIVEN"
    assert classify_price_vs_flow(price_change=0.12, flow=0.08, has_l1=True) == "CONSISTENT"
    assert classify_price_vs_flow(price_change=0.12, flow=None, amount_change=0.35, has_l1=False) == "ACTIVITY_ONLY"
    assert classify_price_vs_flow(price_change=0.12, flow=None, has_l1=False) == "PRICE_ONLY"


def test_allocation_matrix_is_concurrent_evidence_not_transfer() -> None:
    matrix = allocation_evidence_matrix({"equity": -2.0, "gold": 3.0, "bond": 0.0, "cash": 1.0})
    # 股票减少且黄金/现金增加 → 只填并发格，对角为 0
    assert matrix.loc["equity", "gold"] == pytest.approx(2.0)
    assert matrix.loc["equity", "cash"] == pytest.approx(1.0)
    assert matrix.loc["gold", "equity"] == pytest.approx(0.0)
    assert float(matrix.loc["equity", "equity"]) == 0.0


def test_monthly_series_not_upsampled_to_daily(tmp_path: Path) -> None:
    rows = [
        _row("cn_m2", date(2024, 1, 31), 300.0),
        _row("cn_m2", date(2024, 2, 29), 302.0),
        _row("px_csi300", date(2024, 2, 1), 3500.0),
        _row("px_csi300", date(2024, 2, 2), 3510.0),
    ]
    write_long_rows(rows, root=tmp_path)
    staged = read_long_rows(tmp_path)
    m2 = staged[staged["series_id"] == "cn_m2"]
    assert len(m2) == 2
    assert set(pd.to_datetime(m2["date"]).dt.date) == {date(2024, 1, 31), date(2024, 2, 29)}


def test_pipeline_static_rows_and_price_driven_flag(tmp_path: Path) -> None:
    # 黄金 ETF 价格大涨、份额下降 → PRICE_DRIVEN
    days = pd.bdate_range("2024-01-02", periods=21)
    rows: list[FlowRow] = []
    px0, sh0 = 4.0, 100.0
    for i, ts in enumerate(days):
        day = ts.date()
        rows.append(_row("etf_518880_px", day, px0 * (1.01**i)))
        rows.append(_row("etf_518880_shares", day, sh0 - i * 0.2))
        rows.append(_row("etf_518880_amount", day, 1e8 + i))
        rows.append(_row("px_spx", day, 5000 + i))
        rows.append(_row("gold_gld_change", day, -1.0 if i > 10 else 0.5))
    info = run_pipeline(root=tmp_path, rows=rows)
    assert info["fetch"]["n_series"] >= 4
    flow = pd.read_parquet(tmp_path / "diagnostics" / "flow_vs_price.parquet")
    gold = flow[flow["series_id"] == "etf_518880_shares"].iloc[0]
    assert gold["flag"] == "PRICE_DRIVEN"
    real = question_is_real(flow)
    assert "PRICE_DRIVEN" in set(real["flag"])
    changes = pd.read_parquet(tmp_path / "diagnostics" / "changes.parquet")
    src = question_source_dest(changes)
    assert not src.empty
    missing = unavailable_series(changes)
    assert "ici_mmf_aum" in set(missing["series_id"])


def test_change_table_keeps_asof_and_grade(tmp_path: Path) -> None:
    rows = [
        _row("hsgt_north_net", date(2024, 3, 1), 10.0),
        _row("hsgt_north_net", date(2024, 3, 8), -4.0),
        _row("px_dxy", date(2024, 3, 8), 104.0),
        _row("px_dxy", date(2024, 3, 1), 103.0),
    ]
    write_long_rows(rows, root=tmp_path)
    long_df = read_long_rows(tmp_path)
    changes = build_change_table(long_df)
    north = changes[changes["series_id"] == "hsgt_north_net"].iloc[0]
    assert north["evidence_grade"] == "L1"
    assert north["asof"] == "2024-03-08"
    assert north["chg_5d"] == pytest.approx(-14.0) or north["chg_20d"] == pytest.approx(-14.0)


def test_write_compute_outputs_creates_allocation(tmp_path: Path) -> None:
    rows = [
        _row("gold_gld_tonnes", date(2024, 1, 2), 800.0),
        _row("gold_gld_tonnes", date(2024, 1, 22), 820.0),
        _row("hsgt_north_net", date(2024, 1, 2), 5.0),
        _row("hsgt_north_net", date(2024, 1, 22), -8.0),
    ]
    write_long_rows(rows, root=tmp_path)
    summary = write_compute_outputs(read_long_rows(tmp_path), root=tmp_path)
    assert Path(summary["paths"]["allocation"]).is_file()
    alloc = build_allocation_table(read_long_rows(tmp_path))
    assert "disclaimer" in alloc.columns
    assert "不是已证实的资金去向" in str(alloc["disclaimer"].iloc[0])


def test_proxy_error_retries_direct(monkeypatch) -> None:
    from qlib.contrib.data.flow_monitor.adapters.base import safe_call
    from qlib.contrib.data.flow_monitor.adapters.network import reset_network_state_for_tests

    reset_network_state_for_tests()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("FLOW_MONITOR_NO_PROXY", "0")
    calls = {"n": 0}

    def flaky() -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("ProxyError: Unable to connect to proxy")
        return pd.DataFrame({"日期": ["2024-01-02"], "值": [1.0]})

    out = safe_call("probe", flaky)
    assert calls["n"] == 2
    assert not out.empty
    assert "HTTPS_PROXY" not in __import__("os").environ
    reset_network_state_for_tests()


def test_sse_share_dates_skip_weekend() -> None:
    from qlib.contrib.data.flow_monitor.adapters.etf_cn import _sample_days

    # 2026-11-01 是周日
    days = _sample_days(date(2026, 10, 20), date(2026, 11, 1), limit=3)
    assert days
    assert all(day.weekday() < 5 for day in days)
    assert date(2026, 11, 1) not in days
