# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2：价格调整、估算标签、联接不双计、PIT、无龙虎榜不填 0；不打外网。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.data.alloc_v2.adapters.base import row_of
from qlib.contrib.data.alloc_v2.backtest import apply_cost, evaluate_factor, ic_series
from qlib.contrib.data.alloc_v2.etf_metrics import (
    estimated_allocation,
    isolate_universe,
    price_adjusted_flow,
    raw_share_change,
)
from qlib.contrib.data.alloc_v2.factors import (
    assert_no_north_net_in_core,
    build_all_factors,
    lhb_institution_ratio,
    rolling_percentile,
)
from qlib.contrib.data.alloc_v2.mappings import (
    cross_border_codes,
    feeder_codes,
    seed_etf_master,
    write_empty_maps,
)
from qlib.contrib.data.alloc_v2.pipeline import run_pipeline
from qlib.contrib.data.alloc_v2.registry import catalog_rows, core_datasets, dataset_by_id
from qlib.contrib.data.alloc_v2.schema import SERIES_SET, StandardRow
from qlib.contrib.data.alloc_v2.signals import build_signals, pit_slice
from qlib.contrib.data.alloc_v2.store import read_standard, write_standard, rows_to_frame


def _spec(dataset: str):
    return dataset_by_id()[dataset]


def test_catalog_disables_epfr_and_marks_north_net_reference() -> None:
    assert SERIES_SET == "alloc_research@v2"
    lookup = dataset_by_id()
    assert lookup["epfr_fund"].enabled is False
    assert lookup["ici_mmf"].enabled is False
    assert lookup["hsgt_net"].role == "reference_only"
    assert "hsgt_net" not in {d.dataset for d in core_datasets()}
    text = str(catalog_rows())
    assert "EPFR" in text or "epfr" in text.lower()


def test_price_adjusted_flow_subtracts_mark_to_market() -> None:
    aum = pd.Series([100.0, 120.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    price = pd.Series([10.0, 12.0], index=aum.index)
    assert price_adjusted_flow(aum, price).iloc[1] == pytest.approx(0.0)


def test_share_change_estimate_is_tagged_not_cash_flow() -> None:
    shares = pd.DataFrame({"510300": [10.0, 11.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    nav = pd.DataFrame({"510300": [2.0, 2.0]}, index=shares.index)
    raw = raw_share_change(shares)
    est = estimated_allocation(shares, nav)
    assert raw.iloc[1, 0] == pytest.approx(1.0)
    assert est.iloc[1, 0] == pytest.approx(2.0)


def test_feeder_and_cross_border_excluded_from_industry_flow() -> None:
    master = seed_etf_master()
    assert "000051" in feeder_codes(master)
    assert "513100" in cross_border_codes(master)
    frame = pd.DataFrame({"entity_id": ["510300", "000051", "513100"], "value": [1.0, 2.0, 3.0]})
    cleaned = isolate_universe(frame, master=master)
    assert set(cleaned["entity_id"]) == {"510300"}


def test_lhb_missing_stays_na_not_zero() -> None:
    inst = pd.DataFrame({"600000": [np.nan, 1.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    amt = pd.DataFrame({"600000": [np.nan, 5.0]}, index=inst.index)
    ratio = lhb_institution_ratio(inst, amt)
    assert pd.isna(ratio.iloc[0, 0])
    assert ratio.iloc[1, 0] == pytest.approx(0.2)


def test_rolling_percentile_no_lookahead() -> None:
    s = pd.Series(np.arange(10, dtype=float), index=pd.bdate_range("2024-01-02", periods=10))
    pct = rolling_percentile(s, window=5)
    # 最后一点只用此前窗口，不会用到「未来」
    assert pct.iloc[-1] == pytest.approx(1.0)


def test_pit_available_at_filter() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "A"],
            "value": [1.0, 2.0],
            "available_at": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            "factor_id": ["X", "X"],
        }
    )
    sliced = pit_slice(frame, pd.Timestamp("2024-01-15"))
    assert list(sliced["value"]) == [1.0]


def test_north_net_cannot_enter_core_factors() -> None:
    bad = pd.DataFrame({"factor_id": ["HSGT_north_net"], "value": [1.0]})
    with pytest.raises(AssertionError):
        assert_no_north_net_in_core(bad)


def test_pipeline_with_injected_rows(tmp_path: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=30)
    rows: list[StandardRow] = []
    bar = _spec("etf_bar")
    shares = _spec("etf_shares")
    nav = _spec("etf_nav")
    sw = _spec("sw_industry")
    for i, ts in enumerate(days):
        day = ts.date()
        rows.append(row_of(bar, "510300", day, "close", 4.0 * (1.01**i), unit="price"))
        rows.append(row_of(bar, "510300", day, "amount", 1e8 + i, unit="cny"))
        rows.append(row_of(shares, "510300", day, "shares", 100.0 + i, available_at=day + timedelta(days=1), unit="share"))
        rows.append(row_of(nav, "510300", day, "nav", 4.0, available_at=day + timedelta(days=1), unit="nav"))
        rows.append(row_of(bar, "512480", day, "close", 1.0 + i * 0.01, unit="price"))
        rows.append(row_of(shares, "512480", day, "shares", 50.0 + i * 0.5, available_at=day + timedelta(days=1), unit="share"))
        rows.append(row_of(nav, "512480", day, "nav", 1.0, available_at=day + timedelta(days=1), unit="nav"))
        rows.append(row_of(sw, "801080", day, "close", 1000 + i, unit="index"))
        rows.append(row_of(sw, "801080", day, "amount", 1e10 + i, unit="cny"))
        rows.append(row_of(sw, "801780", day, "close", 2000 + i * 0.5, unit="index"))
        rows.append(row_of(sw, "801780", day, "amount", 8e9 + i, unit="cny"))
    # 联接份额不应进入行业流
    rows.append(row_of(shares, "000051", days[-1].date(), "shares", 999.0, unit="share"))
    # 月频不伪装日频：只写两行 M2
    macro = _spec("cn_macro")
    rows.append(row_of(macro, "M2", date(2024, 1, 31), "m2", 300.0, unit="cny"))
    rows.append(row_of(macro, "M2", date(2024, 2, 29), "m2", 302.0, unit="cny"))

    info = run_pipeline(root=tmp_path, rows=rows)
    assert info["fetch"]["n_rows"] > 0
    standard = read_standard(root=tmp_path)
    m2 = standard[(standard["entity_id"] == "M2")]
    assert len(m2) == 2

    factors = build_all_factors(standard)
    assert "Estimated_Allocation" in set(factors["factor_id"])
    assert "Raw_Share_Change" in set(factors["factor_id"])
    # 联接不应出现在 ETF 因子实体中（isolate）
    etf_entities = set(factors.loc[factors["factor_id"] == "Raw_Share_Change", "entity_id"])
    assert "000051" not in etf_entities

    sig = build_signals(factors, standard)
    assert "S1" in sig and "S4" in sig
    assert info["backtest"]["n_results"] >= 1


def test_maps_seed_written(tmp_path: Path) -> None:
    paths = write_empty_maps(tmp_path)
    assert Path(paths["etf_master"]).is_file()
    assert Path(paths["entity_map"]).is_file()


def test_apply_cost_and_ic_helpers() -> None:
    # ic_series 要求单日截面 ≥5 只实体
    idx = pd.bdate_range("2024-01-02", periods=5)
    cols = list("ABCDE")
    factor = pd.DataFrame({c: np.arange(5) + i for i, c in enumerate(cols)}, index=idx, dtype=float)
    forward = pd.DataFrame({c: np.linspace(-0.1, 0.1, 5) * (1 if i % 2 == 0 else -1) for i, c in enumerate(cols)}, index=idx)
    ics = ic_series(factor, forward)
    assert not ics.empty
    gross = pd.Series([0.01, 0.02], index=idx[:2])
    assert apply_cost(gross, cost_bps=10).iloc[0] == pytest.approx(0.01 - 0.002)


def test_evaluate_factor_records_failure_on_empty() -> None:
    result = evaluate_factor(pd.DataFrame(), pd.DataFrame(), factor_id="X")
    assert result["status"] == "failed"


def test_merge_fetch_keeps_history_and_overwrites_key(tmp_path: Path) -> None:
    from qlib.contrib.data.alloc_v2.pipeline import latest_event_date, run_fetch, run_update, UpdateError
    from qlib.contrib.data.alloc_v2.store import read_standard

    bar = _spec("etf_bar")
    old = [
        row_of(bar, "510300", date(2024, 1, 2), "close", 1.0, unit="price"),
        row_of(bar, "510300", date(2024, 1, 10), "close", 2.0, unit="price"),
    ]
    run_fetch(root=tmp_path, rows=old, merge=False)
    assert latest_event_date(tmp_path) == date(2024, 1, 10)

    new = [
        row_of(bar, "510300", date(2024, 1, 10), "close", 2.5, unit="price"),  # 同主键覆盖
        row_of(bar, "510300", date(2024, 1, 15), "close", 3.0, unit="price"),
    ]
    info = run_fetch(root=tmp_path, rows=new, merge=True, start=date(2024, 1, 3), end=date(2024, 1, 15))
    assert info["merged"] is True
    assert info["n_before"] == 2
    staged = read_standard(root=tmp_path)
    closes = staged[(staged["entity_id"] == "510300") & (staged["field"] == "close")].copy()
    closes["event_date"] = pd.to_datetime(closes["event_date"]).dt.date
    by_day = closes.set_index("event_date")["value"].to_dict()
    assert by_day[date(2024, 1, 2)] == pytest.approx(1.0)  # 历史保留
    assert by_day[date(2024, 1, 10)] == pytest.approx(2.5)  # 新值覆盖
    assert by_day[date(2024, 1, 15)] == pytest.approx(3.0)


def test_update_requires_local_standard(tmp_path: Path) -> None:
    from qlib.contrib.data.alloc_v2.pipeline import UpdateError, run_update

    with pytest.raises(UpdateError, match="all --start"):
        run_update(root=tmp_path)


def test_run_update_with_injected_rows(tmp_path: Path) -> None:
    from qlib.contrib.data.alloc_v2.pipeline import run_fetch, run_update
    from qlib.contrib.data.alloc_v2.store import read_standard

    bar = _spec("etf_bar")
    sw = _spec("sw_industry")
    days = pd.bdate_range("2024-01-02", periods=15)
    seed = []
    for i, ts in enumerate(days):
        day = ts.date()
        seed.append(row_of(bar, "510300", day, "close", 4.0 + i * 0.01, unit="price"))
        seed.append(row_of(sw, "801080", day, "close", 1000 + i, unit="index"))
        seed.append(row_of(sw, "801080", day, "amount", 1e10 + i, unit="cny"))
    run_fetch(root=tmp_path, rows=seed, merge=False)

    extra_day = (days[-1] + pd.Timedelta("1D")).date()
    incoming = [
        row_of(bar, "510300", extra_day, "close", 9.9, unit="price"),
        row_of(sw, "801080", extra_day, "close", 2000.0, unit="index"),
        row_of(sw, "801080", extra_day, "amount", 2e10, unit="cny"),
    ]
    summary = run_update(root=tmp_path, overlap_days=7, end=extra_day, rows=incoming)
    assert summary["last_event_date"] == days[-1].date().isoformat()
    assert summary["fetch"]["merged"] is True
    staged = read_standard(root=tmp_path)
    assert len(staged) >= len(seed)
    assert (tmp_path / "_markers" / "update.json").is_file()


def test_labels_display_name_and_captions_cover_dashboard() -> None:
    from qlib.contrib.data.alloc_v2.labels import (
        CHART_CAPTIONS,
        REQUIRED_DASHBOARD_CAPTION_IDS,
        display_name,
        print_caption,
    )

    assert display_name("801080") == "电子"
    assert display_name("SPX") == "标普500"
    assert "综合分" in display_name("IndustryAllocationScore")
    assert display_name("strength") == "宽基强度"
    assert "沪深300" in display_name("510300")
    assert display_name("UNKNOWN_XYZ") == "UNKNOWN_XYZ"
    for chart_id in REQUIRED_DASHBOARD_CAPTION_IDS:
        assert chart_id in CHART_CAPTIONS
        assert "【读图】" in CHART_CAPTIONS[chart_id]
    # 冒烟：未知 id 也有输出
    print_caption("not_exist_chart")


def test_r2_upload_skips_markers_and_uses_prefix(tmp_path: Path, monkeypatch) -> None:
    from qlib.contrib.data.alloc_v2.r2_upload import (
        iter_upload_files,
        logical_key,
        reset_r2_client_for_tests,
        set_r2_uploader,
        upload_tree,
    )

    reset_r2_client_for_tests()
    monkeypatch.setenv("ALLOC_V2_R2_PREFIX", "research/alloc_v2")
    (tmp_path / "standard").mkdir()
    (tmp_path / "maps").mkdir()
    (tmp_path / "factor").mkdir()
    (tmp_path / "research").mkdir()
    (tmp_path / "_markers").mkdir()
    (tmp_path / "standard" / "observations.parquet").write_bytes(b"std")
    (tmp_path / "maps" / "etf_master.parquet").write_bytes(b"map")
    (tmp_path / "factor" / "all.parquet").write_bytes(b"fac")
    (tmp_path / "research" / "signal_s1.parquet").write_bytes(b"sig")
    (tmp_path / "_markers" / "fetch.json").write_text("{}")
    (tmp_path / "standard" / "junk.parquet.tmp").write_bytes(b"tmp")

    rels = [rel for _, rel in iter_upload_files(tmp_path)]
    assert "standard/observations.parquet" in rels
    assert "maps/etf_master.parquet" in rels
    assert "_markers/fetch.json" not in rels
    assert "standard/junk.parquet.tmp" not in rels
    assert logical_key("standard/observations.parquet") == "research/alloc_v2/standard/observations.parquet"

    dry = upload_tree(tmp_path, dry_run=True)
    assert dry["dry_run"] is True
    assert len(dry["keys"]) == 4
    assert all(k.startswith("research/alloc_v2/") for k in dry["keys"])

    put: list[tuple[str, bytes]] = []

    def fake(key: str, payload: bytes) -> None:
        put.append((key, payload))

    set_r2_uploader(fake)
    out = upload_tree(tmp_path, dry_run=False)
    assert not out["failed"]
    assert len(put) == 4
    assert {k for k, _ in put} == set(out["keys"])
    assert (tmp_path / "_markers" / "r2_upload.json").is_file()
    reset_r2_client_for_tests()
