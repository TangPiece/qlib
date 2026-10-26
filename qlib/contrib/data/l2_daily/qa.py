"""当日质检：抽样对账 + 全市场空值率。失败则禁止写盘/上传/进下一日。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import ALL_GROUPS, factor_ids_for_group
from ._prep import prepare_trades
from .csv_io import read_csv_bytes
from .engine import calc_group_for_symbol, list_codes, resolve_code_dir, source_file_path
from .prepare import default_qa_dir
from .util import canonical_symbol, is_equity

TRADE_NULL_MAX = 0.05
BOOK_NULL_MAX = 0.20
GROUP_FAIL_NULL = 0.95
# 缺逐笔成交超过该比例视为坏档，而不是普通停牌。
TRADE_SOURCE_MISS_MAX = 0.25
SAMPLE_CODES = ("600000.SH", "000001.SZ")
COMPARE_FIELDS = ("l2_trade_count", "l2_vwap")


def has_trade_source(raw_root: Path, day: str, code: str) -> bool:
    """是否存在逐笔成交 CSV。停牌占位通常只有空行情。"""
    return source_file_path(raw_root, day, code, "逐笔成交") is not None


def trade_source_miss_rate(raw_root: Path, day: str, codes: list[str]) -> float:
    """当日股票里缺逐笔成交的比例。"""
    if not codes:
        return 1.0
    miss = sum(1 for code in codes if not has_trade_source(raw_root, day, code))
    return miss / len(codes)


def drop_rows_without_trade_source(wide: pd.DataFrame, raw_root: Path, day: str) -> pd.DataFrame:
    """宽表只留有逐笔成交文件的股票，避免停牌空行进 dump/训练。"""
    if wide is None or wide.empty or "symbol" not in wide.columns:
        return wide
    keep = [has_trade_source(raw_root, day, str(code)) for code in wide["symbol"]]
    return wide.loc[keep].reset_index(drop=True)


def group_null_rates(wide: pd.DataFrame) -> dict[str, float]:
    """每组单元格空值率。"""
    out: dict[str, float] = {}
    for group in ALL_GROUPS:
        cols = [c for c in factor_ids_for_group(group) if c in wide.columns]
        if not cols:
            out[group] = 1.0
            continue
        out[group] = float(wide[cols].isna().to_numpy().mean())
    return out


def _close(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=1e-8))


def _prepare_rows(raw_root: Path, day: str, code: str) -> int:
    path = source_file_path(raw_root, day, code, "逐笔成交")
    if path is None:
        return 0
    trades = read_csv_bytes(path.read_bytes(), "逐笔成交")
    return int(len(prepare_trades(trades, code)))


def _sample_dir_codes(raw_day: Path) -> list[str]:
    """SAMPLE_CODES 解析成真实目录名，避免 600000.SH 对不上 600000.sh。"""
    if not raw_day.is_dir():
        return []
    found: list[str] = []
    for sample in SAMPLE_CODES:
        actual = resolve_code_dir(raw_day, sample)
        if actual is not None:
            found.append(actual)
    return found


def _as_row(hit: pd.DataFrame | pd.Series) -> pd.Series:
    """loc 遇到重复 index 会返回 DataFrame，取最后一行。"""
    if isinstance(hit, pd.DataFrame):
        return hit.iloc[-1]
    return hit


def _wide_row(indexed: pd.DataFrame, code: str) -> pd.Series | None:
    """宽表按 symbol 取值，优先规范名，避免命中残留的空 .sh 行。"""
    canon = canonical_symbol(code)
    if canon in indexed.index:
        return _as_row(indexed.loc[canon])
    if code in indexed.index:
        return _as_row(indexed.loc[code])
    target = str(code).lower()
    for key in indexed.index:
        if str(key).lower() == target:
            return _as_row(indexed.loc[key])
    return None


def _index_by_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """用规范化 symbol 做索引，供盘口覆盖比较。"""
    if "symbol" not in frame.columns:
        return frame
    out = frame.copy()
    out["symbol"] = [canonical_symbol(item) if pd.notna(item) else item for item in out["symbol"]]
    return out.set_index("symbol")


def _value_missing(value: Any) -> bool:
    """NaN / 非有限数视为空。"""
    if pd.isna(value):
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def check_day(
    *,
    day: str,
    wide: pd.DataFrame,
    raw_root: Path,
    old_wide: pd.DataFrame | None,
    qa_dir: Path | None = None,
) -> dict[str, Any]:
    """检查一日宽表。返回 {ok, errors, ...}。"""
    errors: list[str] = []
    rates = group_null_rates(wide)
    if rates.get("trade", 1.0) >= GROUP_FAIL_NULL:
        errors.append(f"trade 空值率 {rates['trade']:.3f} ≥ {GROUP_FAIL_NULL}")
    elif rates.get("trade", 1.0) >= TRADE_NULL_MAX:
        errors.append(f"trade 空值率 {rates['trade']:.3f} ≥ {TRADE_NULL_MAX}")
    if rates.get("book", 1.0) >= BOOK_NULL_MAX:
        errors.append(f"book 空值率 {rates['book']:.3f} ≥ {BOOK_NULL_MAX}")

    indexed = _index_by_canonical(wide) if "symbol" in wide.columns else wide
    raw_day = Path(raw_root) / day
    # 读文件用目录真名；宽表查找用规范化代码，兼容 20170510 小写目录。
    codes = _sample_dir_codes(raw_day)
    # 随机样本只抽 A 股，避免 ETF/基金有成交却不进宽表（如 159919.SZ）。
    others = (
        sorted(
            p.name
            for p in raw_day.iterdir()
            if p.is_dir() and p.name not in codes and is_equity(p.name)
        )
        if raw_day.is_dir()
        else []
    )
    rng = np.random.default_rng(int(day) if day.isdigit() else 0)
    extra = [str(x) for x in rng.choice(others, size=min(3, len(others)), replace=False)] if others else []
    samples = []
    for code in codes + extra:
        prep = _prepare_rows(raw_root, day, code)
        live = calc_group_for_symbol(raw_root, day, code, "trade") or {}
        table = _wide_row(indexed, code)
        display = canonical_symbol(code)
        row = {"symbol": display, "prepare_trades": prep}
        for field in COMPARE_FIELDS:
            live_val = live.get(field, float("nan"))
            table_val = table[field] if table is not None and field in table.index else float("nan")
            match = _close(live_val, table_val)
            row[field] = {"live": None if pd.isna(live_val) else float(live_val), "wide": None if pd.isna(table_val) else float(table_val), "match": match}
            live_empty = _value_missing(live_val)
            wide_empty = _value_missing(table_val)
            if prep > 0 and live_empty and wide_empty:
                errors.append(f"{display} {field} 现算/宽表为空，但 prepare_trades={prep}")
            elif prep > 0 and wide_empty:
                errors.append(f"{display} {field} 宽表为空，但 prepare_trades={prep}")
            elif prep > 0 and live_empty:
                errors.append(f"{display} {field} 现算为空，但 prepare_trades={prep}")
            elif prep > 0 and not match:
                errors.append(f"{display} {field} 现算与宽表不一致")
        samples.append(row)

    if old_wide is not None and not old_wide.empty and "l2_obi5_mean" in old_wide.columns and "l2_obi5_mean" in wide.columns:
        old_idx = _index_by_canonical(old_wide) if "symbol" in old_wide.columns else old_wide
        common = indexed.index.intersection(old_idx.index)
        if len(common):
            old_ok = old_idx.loc[common, "l2_obi5_mean"].notna().mean()
            new_ok = indexed.loc[common, "l2_obi5_mean"].notna().mean()
            if old_ok > 0.8 and new_ok < old_ok - 0.2:
                errors.append(f"盘口覆盖从 {old_ok:.3f} 掉到 {new_ok:.3f}")

    payload = {
        "ok": not errors,
        "date": day,
        "n_symbols": int(len(wide)),
        "group_null_rates": rates,
        "samples": samples,
        "errors": errors,
        "trade_source_miss_rate": trade_source_miss_rate(raw_root, day, list_codes(raw_root, day)),
    }
    out_dir = Path(qa_dir) if qa_dir is not None else default_qa_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"l2_daily_day_qa_{day}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload
