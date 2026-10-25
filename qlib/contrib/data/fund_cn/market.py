# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""估值用未复权收盘：默认从 qlib cn_data 的 $close/$factor 还原。"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from .adapters.base import a_code_to_instrument, instrument_to_a_code

_log = logging.getLogger(__name__)

_DEFAULT_QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"


def instrument_to_qlib(instrument_key: str) -> str | None:
    """CNStock:600000 / 600000.SH → SH600000。"""
    text = str(instrument_key or "").strip()
    if not text:
        return None
    if "." in text and text.rsplit(".", 1)[-1].upper() in {"SH", "SZ"}:
        code, exch = text.rsplit(".", 1)
        if code.isdigit():
            return f"{exch.upper()}{code.zfill(6)}"
    code = instrument_to_a_code(text)
    if not code.isdigit() or len(code) != 6:
        return None
    prefix = code[0]
    exch = "SH" if prefix in {"6", "9"} else "SZ"
    return f"{exch}{code}"


def qlib_to_instrument(symbol: str) -> str:
    """SH600000 → CNStock:600000。"""
    text = str(symbol or "").strip().upper()
    if len(text) >= 8 and text[:2] in {"SH", "SZ"} and text[2:].isdigit():
        return a_code_to_instrument(text[2:])
    if "." in text:
        code, _exch = text.rsplit(".", 1)
        return a_code_to_instrument(code)
    return a_code_to_instrument(text)


def _ensure_qlib(qlib_dir: Path) -> None:
    import qlib
    from qlib.config import REG_CN

    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)


def load_market_file(path: str | Path) -> pd.DataFrame:
    """读取显式注入的未复权日行情；CSV/Parquet 共用列契约。"""
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("--market-file 须为 .parquet / .pq / .csv")
    needed = {"instrument_key", "trading_date", "close"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"行情文件缺少列: {sorted(missing)}")
    return frame


def load_unadjusted_close(
    instrument_keys: Sequence[str],
    trading_dates: Sequence[date | str],
    *,
    qlib_dir: str | Path | None = None,
) -> pd.DataFrame:
    """从 qlib 取 $close/$factor，还原 raw_close = close / factor。"""
    from qlib.data import D

    start = min(pd.to_datetime(d) for d in trading_dates)
    end = max(pd.to_datetime(d) for d in trading_dates)
    symbols = [instrument_to_qlib(k) for k in instrument_keys]
    symbols = [s for s in symbols if s]
    if not symbols:
        return pd.DataFrame(columns=["instrument_key", "trading_date", "close"])

    provider = Path(qlib_dir).expanduser() if qlib_dir else _DEFAULT_QLIB_DIR
    try:
        _ensure_qlib(provider)
    except Exception as exc:
        _log.warning("qlib.init 失败 %s: %s", provider, exc)
        return pd.DataFrame(columns=["instrument_key", "trading_date", "close"])

    fields = ["$close"]
    try:
        sample = D.features(symbols[:1], ["$factor"], start_time=start, end_time=end)
        if sample is not None and not sample.empty:
            fields.append("$factor")
    except Exception:
        pass

    panel = D.features(symbols, fields, start_time=start, end_time=end)
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["instrument_key", "trading_date", "close"])

    frame = panel.reset_index()
    # qlib 常见 MultiIndex: instrument, datetime
    if "instrument" in frame.columns:
        frame["instrument_key"] = frame["instrument"].map(qlib_to_instrument)
    elif "symbol" in frame.columns:
        frame["instrument_key"] = frame["symbol"].map(qlib_to_instrument)
    else:
        raise ValueError("qlib features 缺少 instrument/symbol")
    date_col = "datetime" if "datetime" in frame.columns else "date"
    frame["trading_date"] = pd.to_datetime(frame[date_col]).dt.date
    close = pd.to_numeric(frame["$close"], errors="coerce")
    if "$factor" in frame.columns:
        factor = pd.to_numeric(frame["$factor"], errors="coerce")
        raw = close / factor.replace(0, pd.NA)
    else:
        _log.warning("cn_data 无 $factor，按已是未复权 close 使用")
        raw = close
    out = pd.DataFrame(
        {
            "instrument_key": frame["instrument_key"],
            "trading_date": frame["trading_date"],
            "close": raw,
        }
    )
    wanted = {pd.to_datetime(d).date() for d in trading_dates}
    return out[out["trading_date"].isin(wanted)].reset_index(drop=True)


def default_instruments(*, qlib_dir: str | Path | None = None) -> list[str]:
    """默认用 qlib all 市场代码，转成 CNStock:xxxxxx。"""
    from qlib.data import D

    provider = Path(qlib_dir).expanduser() if qlib_dir else _DEFAULT_QLIB_DIR
    _ensure_qlib(provider)
    # instruments("all") 返回市场配置，list_instruments 才展开代码
    raw = D.instruments("all")
    codes = D.list_instruments(raw, as_list=True)
    return [qlib_to_instrument(c) for c in codes]


def calendar_dates(start: str, end: str, *, qlib_dir: str | Path | None = None) -> list[str]:
    """交易日历 YYYY-MM-DD。"""
    from qlib.data import D

    provider = Path(qlib_dir).expanduser() if qlib_dir else _DEFAULT_QLIB_DIR
    _ensure_qlib(provider)
    days = D.calendar(start_time=start, end_time=end, freq="day")
    return [pd.Timestamp(d).date().isoformat() for d in days]
