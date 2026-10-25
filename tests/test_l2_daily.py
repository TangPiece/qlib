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
