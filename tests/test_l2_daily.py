# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L2 日频流水线单测：列名 strip、全 NaN 记 fail。"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from qlib.contrib.data.l2_daily.csv_io import normalize, read_csv_bytes
from qlib.contrib.data.l2_daily.engine import _append_symbol_result, _group_all_nan
from qlib.contrib.data.l2_daily.groups.trade import calc_trade_factors
from qlib.contrib.data.l2_daily._prep import prepare_trades


def test_normalize_strips_leading_spaces_in_headers() -> None:
    """2018 万得 CSV 逗号后带空格，列名必须 strip 才能被 prepare_trades 识别。"""
    raw = pd.DataFrame(
        {
            " 时间": ["093000000"],
            " 成交价格": ["100000"],
            " 成交数量": ["100"],
            " 成交代码": ["0"],
            " BS标志": ["B"],
            " 叫买序号": ["1"],
            " 叫卖序号": ["2"],
        }
    )
    frame = normalize(raw, "逐笔成交")
    assert "时间" in frame.columns
    assert "成交价格" in frame.columns
    assert " 时间" not in frame.columns


def test_read_csv_bytes_strips_spaced_headers() -> None:
    """gb18030 CSV 表头含前导空格时，读入后列名应干净。"""
    text = " 时间, 成交价格, 成交数量, 成交代码, BS标志, 叫买序号, 叫卖序号\n093000001,100000,200,0,B,1,2\n"
    frame = read_csv_bytes(text.encode("gb18030"), "逐笔成交")
    assert list(frame.columns)[:3] == ["时间", "成交价格", "成交数量"]
    prepared = prepare_trades(frame, "600000.SH")
    assert len(prepared) == 1


def test_group_all_nan_is_fail() -> None:
    """组内有效列全是 NaN 不能标 done。"""
    columns = ["l2_trade_count", "l2_vwap"]
    assert _group_all_nan({"l2_trade_count": float("nan"), "l2_vwap": float("nan")}, columns)
    assert not _group_all_nan({"l2_trade_count": 10937.0, "l2_vwap": 10.1}, columns)

    rows: list[dict] = []
    status_rows: list[dict] = []
    ok, fail = _append_symbol_result(
        trade_date="20180731",
        code="600000.SH",
        factor_group="trade",
        result={"l2_trade_count": float("nan"), "l2_vwap": float("nan")},
        columns=columns,
        rows=rows,
        status_rows=status_rows,
    )
    assert (ok, fail) == (0, 1)
    assert status_rows[0]["status"] == "failed"
    assert rows == []


def test_empty_trades_are_all_nan_fail() -> None:
    """成交明细为空时 trade 组返回全 NaN，编排应记 fail。"""
    result = calc_trade_factors(pd.DataFrame(), "600000.SH")
    columns = [k for k in result if k.startswith("l2_")]
    assert columns
    assert _group_all_nan(result, columns)
    assert math.isnan(float(result["l2_trade_count"]))


def test_missing_trade_source_is_skipped_not_failed(tmp_path: Path) -> None:
    """没有逐笔成交文件时记 skipped，不抬失败率。"""
    from qlib.contrib.data.l2_daily.engine import _append_symbol_result

    day = tmp_path / "20170104" / "000029.SZ"
    day.mkdir(parents=True)
    (day / "行情.csv").write_text("x", encoding="utf-8")
    rows: list[dict] = []
    status_rows: list[dict] = []
    ok, fail = _append_symbol_result(
        trade_date="20170104",
        code="000029.SZ",
        factor_group="trade",
        result=None,
        columns=["l2_trade_count"],
        rows=rows,
        status_rows=status_rows,
        books_root=tmp_path,
    )
    assert (ok, fail) == (0, 0)
    assert rows == []
    assert status_rows[0]["status"] == "skipped"


def test_drop_rows_without_trade_source(tmp_path: Path) -> None:
    """宽表丢掉缺逐笔成交的股票；缺源率按目录统计。"""
    from qlib.contrib.data.l2_daily.qa import (
        drop_rows_without_trade_source,
        trade_source_miss_rate,
    )

    day = tmp_path / "20170104"
    ok_dir = day / "000001.SZ"
    miss_dir = day / "000029.SZ"
    ok_dir.mkdir(parents=True)
    miss_dir.mkdir()
    (ok_dir / "逐笔成交.csv").write_text("t", encoding="utf-8")
    (miss_dir / "行情.csv").write_text("q", encoding="utf-8")
    wide = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000029.SZ"],
            "l2_trade_count": [10.0, float("nan")],
        }
    )
    kept = drop_rows_without_trade_source(wide, tmp_path, "20170104")
    assert list(kept["symbol"]) == ["000001.SZ"]
    assert trade_source_miss_rate(tmp_path, "20170104", ["000001.SZ", "000029.SZ"]) == 0.5


def test_dropna_cols_drops_missing_trade_count() -> None:
    """Fillna 之前丢掉 l2_trade_count 为空的行。"""
    from qlib.data.dataset.processor import DropnaCols

    frame = pd.DataFrame(
        {
            "l2_trade_count": [10.0, float("nan"), 3.0],
            "l2_vwap": [1.0, 2.0, float("nan")],
        }
    )
    out = DropnaCols(col_list=["l2_trade_count"])(frame)
    assert list(out.index) == [0, 2]
    multi = pd.DataFrame(
        [[10.0, 1.0], [float("nan"), 2.0]],
        columns=pd.MultiIndex.from_tuples([("feature", "l2_trade_count"), ("feature", "l2_vwap")]),
    )
    out_multi = DropnaCols(col_list=["l2_trade_count"])(multi)
    assert len(out_multi) == 1


def test_delete_day_sources_keep_archive(tmp_path: Path) -> None:
    """失败只删解压目录；通过才连 7z 一起删。"""
    from qlib.contrib.data.l2_daily.prepare import delete_day_sources

    day_dir = tmp_path / "20220601"
    archive = tmp_path / "20220601.7z"
    day_dir.mkdir()
    (day_dir / "x.csv").write_text("ok", encoding="utf-8")
    archive.write_bytes(b"7z")

    removed = delete_day_sources("20220601", tmp_path, keep_archive=True)
    assert str(day_dir) in removed
    assert not day_dir.exists()
    assert archive.is_file()

    day_dir.mkdir()
    (day_dir / "x.csv").write_text("ok", encoding="utf-8")
    removed = delete_day_sources("20220601", tmp_path, keep_archive=False)
    assert str(archive) in removed
    assert not archive.exists()
    assert not day_dir.exists()
