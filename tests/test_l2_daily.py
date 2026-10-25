# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L2 日频流水线单测：列名 strip、全 NaN 记 fail。"""

from __future__ import annotations

import json
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


def _write_gb18030(path: Path, text: str) -> None:
    """写入万得同款 gb18030 CSV，供计算单测使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("gb18030"))


def test_source_file_path_ignores_parquet_only(tmp_path: Path) -> None:
    """只有 Parquet、没有 CSV 时视为缺源。"""
    from qlib.contrib.data.l2_daily.engine import source_file_path
    from qlib.contrib.data.l2_daily.prepare import day_has_sources

    code_dir = tmp_path / "20170116" / "000001.SZ"
    code_dir.mkdir(parents=True)
    (code_dir / "逐笔成交.parquet").write_bytes(b"parq")
    assert source_file_path(tmp_path, "20170116", "000001.SZ", "逐笔成交") is None
    assert not day_has_sources(tmp_path, "20170116")


def test_calc_groups_for_symbol_reads_each_file_once(tmp_path: Path, monkeypatch) -> None:
    """四组同算时每类 CSV 只读一次，且四组都有结果。"""
    from qlib.contrib.data.l2_daily import engine
    from qlib.contrib.data.l2_daily.catalog import ALL_GROUPS

    code_dir = tmp_path / "20170116" / "000001.SZ"
    _write_gb18030(
        code_dir / "逐笔成交.csv",
        "时间,成交价格,成交数量,成交代码,BS标志,叫买序号,叫卖序号\n"
        "093000001,100000,200,0,B,1,2\n"
        "093100001,100100,100,0,S,3,4\n",
    )
    _write_gb18030(
        code_dir / "行情.csv",
        "时间,申买价1,申卖价1,申买量1,申卖量1\n093100000,100000,100200,100,80\n",
    )
    _write_gb18030(
        code_dir / "逐笔委托.csv",
        "时间,交易所委托号,委托类型,委托代码,委托价格,委托数量\n093000000,1,A,1,100000,200\n",
    )

    reads: list[str] = []
    original = engine._read_path

    def _count_read(path, ftype):
        reads.append(ftype)
        return original(path, ftype)

    monkeypatch.setattr(engine, "_read_path", _count_read)
    results = engine.calc_groups_for_symbol(tmp_path, "20170116", "000001.SZ", list(ALL_GROUPS))
    assert set(results) == set(ALL_GROUPS)
    assert results["trade"] is not None
    assert results["trade"]["l2_trade_count"] == 2
    assert set(reads) == {"行情", "逐笔成交", "逐笔委托"}
    assert len(reads) == 3


def test_run_one_day_recycles_csv_dir_on_fail(tmp_path: Path) -> None:
    """计算/质检失败也要删解压目录，只留 7z。"""
    from qlib.contrib.data.l2_daily.pipeline import run_one_day

    day = "20170116"
    code_dir = tmp_path / day / "000001.SZ"
    _write_gb18030(
        code_dir / "行情.csv",
        "时间,申买价1,申卖价1,申买量1,申卖量1\n093100000,100000,100200,100,80\n",
    )
    archive = tmp_path / f"{day}.7z"
    archive.write_bytes(b"7z")
    out = tmp_path / "out"
    summary = run_one_day(
        day,
        books_root=tmp_path,
        output_root=out,
        workers=1,
        upload_r2=False,
        compact_months=False,
        delete_sources=True,
    )
    assert int(summary["failed"]) > 0
    assert not (tmp_path / day).exists()
    assert archive.is_file()
    assert any(str(tmp_path / day) in item for item in summary.get("deleted") or [])


def test_load_r2_env_fills_empty_keys_only(tmp_path: Path, monkeypatch) -> None:
    """本仓库 .env 只填空键，不覆盖已有环境变量。"""
    import os

    from qlib.contrib.data.l2_daily import r2_env

    env = tmp_path / ".env"
    env.write_text(
        "R2_ENDPOINT_URL=https://example.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=from-file\n"
        "R2_SECRET_ACCESS_KEY=secret\n"
        "R2_BUCKET=level2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("R2_ENV_FILE", str(env))
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "keep-me")
    r2_env._LOADED = None
    assert r2_env.load_r2_env() == env
    assert os.environ["R2_ENDPOINT_URL"].startswith("https://example")
    assert os.environ["R2_ACCESS_KEY_ID"] == "keep-me"
    assert os.environ["R2_BUCKET"] == "level2"


def test_upload_day_to_r2_none_when_unconfigured(tmp_path: Path, monkeypatch) -> None:
    """未配置时不假装 uploaded。"""
    from qlib.contrib.data.l2_daily import storage

    monkeypatch.setenv("R2_ENV_FILE", str(tmp_path / "missing.env"))
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    storage.set_r2_uploader(None)
    storage._client = None
    path = storage.day_path(tmp_path, "20170103")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet")
    assert storage.upload_day_to_r2(tmp_path, "20170103") is None


def test_parse_reupload_dates_from_local_factors(tmp_path: Path) -> None:
    """reupload --start/--end 按本地宽表筛日期，不看 raw。"""
    import argparse

    from qlib.contrib.data.l2_daily import storage
    from qlib.contrib.data.l2_daily.__main__ import _parse_explicit_dates

    for day in ("20170103", "20170313", "20170401"):
        path = storage.day_path(tmp_path, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    args = argparse.Namespace(
        dates="",
        start="20170101",
        end="20170313",
        output=str(tmp_path),
        factor_version=1,
        raw_dir="",
    )
    assert _parse_explicit_dates(args, from_factors=True) == ["20170103", "20170313"]


def test_publish_day_marks_r2_failure_without_raise(tmp_path: Path) -> None:
    """上传抛错时本地宽表已写、账本有该日、不往上抛。"""
    from qlib.contrib.data.l2_daily import publish, storage
    from qlib.contrib.data.l2_daily.catalog import wide_columns

    def _boom(_key: str, _payload: bytes) -> None:
        raise RuntimeError("r2 down")

    storage.set_r2_uploader(_boom)
    try:
        frame = pd.DataFrame([{col: ("20170103" if col == "trade_date" else "000001.SZ" if col == "symbol" else 1.0) for col in wide_columns()}])
        result = publish.publish_day(tmp_path, "20170103", frame)
        assert result["r2_failed"] is True
        assert result["uri"] is None
        assert storage.day_path(tmp_path, "20170103").is_file()
        assert storage.list_r2_failed(tmp_path) == ["20170103"]
    finally:
        storage.set_r2_uploader(None)


def test_run_continues_after_r2_upload_failure(tmp_path: Path, monkeypatch) -> None:
    """run() 两天时第一天 R2 失败不阻断第二天计算。"""
    from qlib.contrib.data.l2_daily import pipeline, storage

    seen: list[str] = []

    def _fake_day(trade_date: str, **kwargs):
        # 只验证编排：第一天 R2 失败不应 break 第二天。
        seen.append(trade_date)
        if trade_date == "20170103":
            return {"processed": 1, "skipped": 0, "failed": 0, "deleted": [], "r2_failed": ["20170103"]}
        return {"processed": 1, "skipped": 0, "failed": 0, "deleted": [], "r2_failed": []}

    monkeypatch.setattr(pipeline, "run_one_day", _fake_day)
    summary = pipeline.run(
        ["20170103", "20170104"],
        books_root=tmp_path,
        output_root=tmp_path / "out",
        upload_r2=True,
        compact_months=False,
        delete_sources=False,
    )
    assert seen == ["20170103", "20170104"]
    assert summary["failed"] == 0
    assert summary["r2_failed"] == ["20170103"]
    assert int(summary["processed"]) == 2


def test_reupload_failed_retries_and_clears(tmp_path: Path) -> None:
    """reupload --failed 只选账本日期；成功清标记，再失败则保留并继续后面的天。"""
    from qlib.contrib.data.l2_daily import storage
    from qlib.contrib.data.l2_daily.__main__ import main

    for day in ("20170103", "20170104", "20170401"):
        path = storage.day_path(tmp_path, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")
    storage.mark_r2_failed(tmp_path, "20170103", "timeout")
    storage.mark_r2_failed(tmp_path, "20170104", "timeout")

    uploaded: list[str] = []

    def _fail_first(key: str, _payload: bytes) -> None:
        uploaded.append(key)
        if "20170103" in key:
            raise RuntimeError("still down")

    storage.set_r2_uploader(_fail_first)
    try:
        code = main(
            [
                "reupload",
                "--failed",
                "--output",
                str(tmp_path),
                "--keep-old-r2",
            ]
        )
        assert code == 1
        assert storage.list_r2_failed(tmp_path) == ["20170103"]
        assert any("20170103" in item for item in uploaded)
        assert any("20170104" in item for item in uploaded)
        assert not any("20170401" in item for item in uploaded)
    finally:
        storage.set_r2_uploader(None)


def test_discover_downloading_dates(tmp_path: Path) -> None:
    """识别百度云/aria2/part，忽略完整 7z。"""
    from qlib.contrib.data.l2_daily.prepare import discover_downloading_dates

    (tmp_path / "20170410.7z.baiduyun.p.downloading").write_text("x", encoding="utf-8")
    (tmp_path / "20170411.7z.aria2").write_text("x", encoding="utf-8")
    (tmp_path / "20170412.7z.part").write_text("x", encoding="utf-8")
    (tmp_path / "20170413.7z").write_text("z", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("n", encoding="utf-8")
    assert discover_downloading_dates(tmp_path) == ["20170410", "20170411", "20170412"]


def test_run_batch_waits_for_download(tmp_path: Path, monkeypatch) -> None:
    """只有 downloading 时不结束，变成 .7z 后进入 planned。"""
    from qlib.contrib.data.l2_daily import pipeline, prepare

    raw = tmp_path / "raw"
    raw.mkdir()
    marker = raw / "20170410.7z.baiduyun.p.downloading"
    marker.write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    slept = {"n": 0}

    def _sleep(_seconds: float) -> None:
        slept["n"] += 1
        marker.unlink()
        (raw / "20170410.7z").write_text("z", encoding="utf-8")

    monkeypatch.setenv("LEVEL2_DOWNLOAD_POLL_SEC", "1")
    monkeypatch.setattr(pipeline.time, "sleep", _sleep)
    monkeypatch.setattr(prepare, "ensure_day_csv", lambda *args, **kwargs: raw / "20170410")
    monkeypatch.setattr(
        pipeline,
        "run_one_day",
        lambda *args, **kwargs: {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        },
    )
    summary = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert slept["n"] >= 1
    assert summary["dates"] == ["20170410"]
    assert summary["failed"] == 0


def test_run_batch_qa_fail_continues_and_skips_ledger(tmp_path: Path, monkeypatch) -> None:
    """质检失败记账后继续下一天；下次 all 跳过账本日期。"""
    from qlib.contrib.data.l2_daily import pipeline, prepare, storage

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "20170104.7z").write_text("z", encoding="utf-8")
    (raw / "20170105.7z").write_text("z", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(prepare, "ensure_day_csv", lambda *args, **kwargs: raw)

    def _run(day: str, **kwargs):
        if day == "20170104":
            storage.mark_qa_failed(out, day, "trade 空值率")
            return {
                "processed": 1,
                "skipped": 0,
                "failed": 1,
                "qa_failed": True,
                "error": "trade 空值率",
                "deleted": [],
                "r2_failed": [],
            }
        return {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        }

    monkeypatch.setattr(pipeline, "run_one_day", _run)
    first = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert first["dates"] == ["20170104", "20170105"]
    assert first["qa_failed"] == ["20170104"]
    assert storage.list_qa_failed(out) == ["20170104"]

    ran: list[str] = []

    def _run_again(day: str, **kwargs):
        ran.append(day)
        return {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        }

    monkeypatch.setattr(pipeline, "run_one_day", _run_again)
    second = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert ran == ["20170105"]
    assert "20170104" not in second["dates"]


def test_list_day_failed_missing_kind_is_qa(tmp_path: Path) -> None:
    """旧账本没有 kind 时按质检失败，仍会被 all 跳过。"""
    from qlib.contrib.data.l2_daily import storage

    out = tmp_path / "out"
    out.mkdir()
    (out / storage.QA_FAILED_NAME).write_text(
        '{"20170104": {"error": "trade 空值率", "updated_at": "2026-01-01T00:00:00+08:00"}}',
        encoding="utf-8",
    )
    assert storage.list_day_failed(out) == ["20170104"]
    assert storage.list_day_failed(out, kind="qa") == ["20170104"]
    assert storage.list_qa_failed(out) == ["20170104"]
    assert storage.list_day_failed(out, kind="extract") == []


def test_run_batch_extract_fail_continues_and_skips_ledger(tmp_path: Path, monkeypatch) -> None:
    """解压失败记账后继续下一天；下次 all 跳过账本日期。"""
    from qlib.contrib.data.l2_daily import pipeline, prepare, storage

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "20170104.7z").write_text("z", encoding="utf-8")
    (raw / "20170105.7z").write_text("z", encoding="utf-8")
    out = tmp_path / "out"

    def _ensure(day: str, *args, **kwargs):
        if day == "20170104":
            raise RuntimeError("7z broken")
        return raw / day

    monkeypatch.setattr(prepare, "ensure_day_csv", _ensure)
    monkeypatch.setattr(
        pipeline,
        "run_one_day",
        lambda day, **kwargs: {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        },
    )
    first = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert first["dates"] == ["20170104", "20170105"]
    assert first["extract_failed"] == ["20170104"]
    assert first["failed"] == 1
    assert storage.list_day_failed(out) == ["20170104"]
    assert storage.list_day_failed(out, kind="extract") == ["20170104"]
    ledger = json.loads(storage.qa_failed_path(out).read_text(encoding="utf-8"))
    assert ledger["20170104"]["kind"] == "extract"

    ran: list[str] = []

    def _run_again(day: str, **kwargs):
        ran.append(day)
        return {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        }

    monkeypatch.setattr(pipeline, "run_one_day", _run_again)
    second = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert ran == ["20170105"]
    assert "20170104" not in second["dates"]


def test_run_batch_compute_fail_continues_and_skips_ledger(tmp_path: Path, monkeypatch) -> None:
    """计算未捕获异常记账后继续下一天；下次 all 跳过账本日期。"""
    from qlib.contrib.data.l2_daily import pipeline, prepare, storage

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "20170104.7z").write_text("z", encoding="utf-8")
    (raw / "20170105.7z").write_text("z", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(prepare, "ensure_day_csv", lambda *args, **kwargs: raw)

    def _run(day: str, **kwargs):
        if day == "20170104":
            raise RuntimeError("pool crashed")
        return {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        }

    monkeypatch.setattr(pipeline, "run_one_day", _run)
    first = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert first["dates"] == ["20170104", "20170105"]
    assert first["compute_failed"] == ["20170104"]
    assert first["failed"] == 1
    assert storage.list_day_failed(out) == ["20170104"]
    assert storage.list_day_failed(out, kind="compute") == ["20170104"]
    ledger = json.loads(storage.qa_failed_path(out).read_text(encoding="utf-8"))
    assert ledger["20170104"]["kind"] == "compute"

    ran: list[str] = []

    def _run_again(day: str, **kwargs):
        ran.append(day)
        return {
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
        }

    monkeypatch.setattr(pipeline, "run_one_day", _run_again)
    second = pipeline.run_batch(raw_root=raw, output_root=out, workers=1, delete_sources=False)
    assert ran == ["20170105"]
    assert "20170104" not in second["dates"]


def test_run_one_day_group_fail_records_compute(tmp_path: Path, monkeypatch) -> None:
    """整组计算失败记 kind=compute，删解压留 7z。"""
    from qlib.contrib.data.l2_daily import pipeline, storage

    day = "20170116"
    code_dir = tmp_path / day / "000001.SZ"
    code_dir.mkdir(parents=True)
    (code_dir / "行情.csv").write_text("时间\n093100000\n", encoding="utf-8")
    archive = tmp_path / f"{day}.7z"
    archive.write_bytes(b"7z")
    out = tmp_path / "out"
    monkeypatch.setattr(
        pipeline,
        "calc_groups_day",
        lambda *args, **kwargs: {"trade": ([], [], 0, 10)},
    )
    summary = pipeline.run_one_day(
        day,
        books_root=tmp_path,
        output_root=out,
        groups=["trade"],
        workers=1,
        upload_r2=False,
        compact_months=False,
        delete_sources=True,
    )
    assert summary["fail_kind"] == "compute"
    assert int(summary["failed"]) == 1
    assert not (tmp_path / day).exists()
    assert archive.is_file()
    assert storage.list_day_failed(out, kind="compute") == [day]
